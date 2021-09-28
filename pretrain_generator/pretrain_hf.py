import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import collections
import random
import torchcontrib

from electra_pretrain_model import ElectraGenerator
import tqdm
import pdb

class ELECTRATrainer:
    """
    ELECTRATrainer make the pretrained ELECTRA model 
    """

    def __init__(self, electra: ElectraGenerator, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 100, log_file=None,append=False, momentum=0.9, grad_acc_steps=1):
        """
        :param electra: ELECTRA model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """
        # Setup cuda device for ELECTRA training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.hardware = "cuda" if cuda_condition else "cpu"

        # This ELECTRA model will be saved every epoch
        self.electra = electra.to(self.device)
        self.electra = self.electra.float()


        #pdb.set_trace()
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for ELECTRA" % torch.cuda.device_count())
            self.electra = nn.DataParallel(self.electra, device_ids=cuda_devices)
            self.hardware = "parallel"
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        #self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        #self.optim_schedule = ScheduledOptim(self.optim, self.electra.hidden, n_warmup_steps=warmup_steps)
        #base_optim = SGD(self.electra.parameters(),lr=lr,momentum=momentum)
        #self.optim = torchcontrib.optim.SWA(base_optim, swa_start=10, swa_freq=5, swa_lr=0.05)
        self.optim = SGD(self.electra.parameters(),lr=lr,momentum=momentum)
        #self.optim = Adam(self.electra.parameters(),lr=lr,eps=1e-06)

        self.grad_acc_steps = grad_acc_steps

        self.log_freq = log_freq

        # clear log file
        if log_file:
            self.log_file = log_file
            if not append:
                with open(self.log_file,"w+") as f:
                    f.write("EPOCH,MODE,AVG LOSS,TOTAL CORRECT,TOTAL ELEMENTS,ACCURACY,MASK CORRECT,MASK 3 CORRECT,MASK 5 CORRECT,MASK 10 CORRECT,TOTAL MASK,MASK ACCURACY\n")
        print("Total Parameters:", sum([p.nelement() for p in self.electra.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"



        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        #pdb.set_trace()
        cumulative_g_loss = 0.0

        g_total_correct = 0
        g_total_mask_correct = 0
        g_total_3_mask_correct = 0
        g_total_5_mask_correct = 0
        g_total_10_mask_correct = 0
        total_element = 0
        total_mask = 0

        tokens_masked = []
        tokens_predicted = []
        times_predicted_present_microbes = 0

        for i, data in data_iter:
            #if random.random() < 0.9:
            #    continue
 
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            #create attention mask
            zero_boolean = torch.eq(data["species_frequencies"],0)
            mask = torch.ones(zero_boolean.shape,dtype=torch.float).to(self.device)
            mask = mask.masked_fill(zero_boolean,0)

            #change label for non-masked tokens to -100 so generator ignores predictions on non-masked tokens
            data["electra_mask_label"] = data["electra_label"].masked_fill(~data["mask_locations"],-100) 
            
            #for w in range(0, len(data["electra_input"])):
            #    if data["electra_input"][0][w] in [26728, 26726, 26727]:
            #        continue
            #    if data["electra_input"][0][w] in data["electra_input"][0][w:]:
            #        print("Repeat found:", w, data["electra_input"][0])

            #print(data["electra_input"])
            g_loss,g_scores = self.electra.forward(data["electra_input"],mask,data["electra_mask_label"])             
            # 3. backward and optimization only in train
            if train:
                #self.optim_schedule.zero_grad()
                if i % self.grad_acc_steps == 0:
                    self.optim.zero_grad()
                if self.hardware == "parallel":
                    #pdb.set_trace()
                    g_loss.mean().backward()
                else:
                    g_loss.backward()
                #self.optim_schedule.step_and_update_lr()
                if i % self.grad_acc_steps == 0:
                    self.optim.step()
            #pdb.set_trace()

            #get generator accuracy for masked tokens
            g_predictions = g_scores.sort(2,descending=True)[1]
            g_mask_predictions = torch.masked_select(g_predictions,data["mask_locations"].unsqueeze(2).expand(g_predictions.shape[0],g_predictions.shape[1],g_predictions.shape[2])).reshape((data["mask_locations"].sum().item(),g_predictions.shape[2]))
            #print(g_predictions[:, :, 0], data["mask_locations"].size())
            a = torch.masked_select(g_predictions[0, :, 0], data["mask_locations"][0])
            a_l = torch.masked_select(data["electra_label"][0], ~data["mask_locations"][0])

            b = torch.masked_select(g_predictions[1, :, 0], data["mask_locations"][1])
            b_l = torch.masked_select(data["electra_label"][1], ~data["mask_locations"][1])

            #print('a:   ', a, '\na_l: ', a_l)
            mask_locs_a = torch.masked_select(torch.tensor([iloc for iloc in range(513)]).cuda(0), data["mask_locations"][0])
            non_mask_locs_a = torch.masked_select(torch.tensor([iloc for iloc in range(513)]).cuda(0), ~data["mask_locations"][0])

            mask_locs_b = torch.masked_select(torch.tensor([iloc for iloc in range(513)]).cuda(0), data["mask_locations"][1])
            non_mask_locs_b = torch.masked_select(torch.tensor([iloc for iloc in range(513)]).cuda(0), ~data["mask_locations"][1])

            batch_times_predicted_present_microbes = 0
            end_of_a = 0
            end_of_b = 0
            pred_a = g_predictions[0, :, 0]
            pred_b = g_predictions[1, :, 0]
            print_out = random.random() < -0.02
            if print_out:
                print('a:   ', a, '\na_l: ', a_l)
            for mask_token_num in mask_locs_a:
                if pred_a[mask_token_num] in a_l:
                    batch_times_predicted_present_microbes += 1
                    if print_out:
                        print('tok:', pred_a[mask_token_num].item(), 'loc:', mask_token_num.item())
                #for input_token_num in non_mask_locs_a:
                #    if data["electra_label"][0][input_token_num] == -100:
                #        break
                #    if input_token_num != mask_token_num and pred_a[mask_token_num] == data["electra_label"][0][input_token_num]:
                #        batch_times_predicted_present_microbes += 1
                    #print((mask_token_num.item(), g_predictions[0, :, 0][mask_token_num].item()))
                    #print("   ", (a_l == g_predictions[0, :, 0][mask_token_num].item()).nonzero(as_tuple=True))

            #print('b:', b, '\n  ', b_l)
            for mask_token_num in mask_locs_b:
                if pred_b[mask_token_num] in b_l:
                    batch_times_predicted_present_microbes += 1
                #for input_token_num in non_mask_locs_b:
                #    if data["electra_label"][1][input_token_num] == -100:
                #        break
                #    if input_token_num != mask_token_num and pred_b[mask_token_num] == data["electra_label"][1][input_token_num]:
                #        batch_times_predicted_present_microbes += 1
                        #print('b', input_token_num, mask_token_num)


            times_predicted_present_microbes += batch_times_predicted_present_microbes



            g_mask_token_labels = torch.masked_select(data["electra_mask_label"],data["mask_locations"])
            g_total_mask_correct += torch.sum(g_mask_predictions[:,0] == g_mask_token_labels).item()
            g_total_3_mask_correct += ELECTRATrainer.check_top_x_mask_predictions(g_mask_predictions,g_mask_token_labels,3)
            g_total_5_mask_correct += ELECTRATrainer.check_top_x_mask_predictions(g_mask_predictions,g_mask_token_labels,5)
            g_total_10_mask_correct += ELECTRATrainer.check_top_x_mask_predictions(g_mask_predictions,g_mask_token_labels,10)        
            total_mask += g_mask_token_labels.shape[0]
            #print(g_mask_token_labels) 


            #for token_pred in g_mask_predictions[:,0]:
            #    tokens_predicted.append(str(token_pred.item()))
            #for token_masked in g_mask_token_labels:
            #    tokens_masked.append(str(token_masked.item()))

            #batch_times_predicted_present_microbes = 0
            #for pred_token_num in range(len(g_mask_token_labels)):
            #    for mask_token_num in range(len(g_mask_token_labels)):
            #        if pred_token_num != mask_token_num and g_mask_predictions[:,0][pred_token_num] == g_mask_token_labels[mask_token_num]:
            #            batch_times_predicted_present_microbes += 1
            #            print(pred_token_num, mask_token_num)
            #times_predicted_present_microbes += batch_times_predicted_present_microbes
 
            #print(g_mask_predictions[:,0])
            #print(batch_times_predicted_present_microbes)

            del g_mask_predictions
            del g_mask_token_labels

            #get generator accuracy for all tokens
            g_total_correct += torch.masked_select((g_predictions[:,:,0] == data["electra_label"]),mask.bool()).sum().item()


            del g_predictions
            del g_scores

            total_element += mask.sum().item()


            log_loss = 0
            if self.hardware == "parallel":
                log_loss = g_loss.sum().item()
                cumulative_g_loss += g_loss.sum().item()


            else:
                log_loss = g_loss.item()    
                cumulative_g_loss += g_loss.item()

            if i % self.log_freq == 0:
                data_iter.write("epoch: {}, iter: {}, avg loss: {},accuracy: {}/{}={:.2f}%, mask accuracy: {}/{}={:.2f}%, loss: {}".format(epoch,i,cumulative_g_loss/(i+1),g_total_correct,total_element,g_total_correct/total_element*100,g_total_mask_correct,total_mask,g_total_mask_correct/total_mask*100,log_loss))

 
            del data
            del mask
            del g_loss



        print("EP{}_{}, avg_loss={}, accuracy={:.2f}%".format(epoch,str_code,cumulative_g_loss /(len(data_iter)*data_loader.batch_size),g_total_mask_correct/total_mask*100))
        if self.log_file:
            with open(self.log_file,"a") as f:
                f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,str_code,cumulative_g_loss/(len(data_loader)*data_loader.batch_size),g_total_correct,total_element,g_total_correct/total_element*100,g_total_mask_correct,g_total_3_mask_correct,g_total_5_mask_correct,g_total_10_mask_correct,total_mask,g_total_mask_correct/total_mask*100))
        
        #self.optim.swap_swa_sgd()
        print("times_predicted_present_microbes:", times_predicted_present_microbes)
        #mask_count = collections.Counter(tokens_masked)
        #pred_count = collections.Counter(tokens_predicted)
        #for t in tokens_masked:
        #    mask_count[i] = mask_count.get(i, 1) + 1
        #for t in tokens_predicted:
        #    pred_count[i] = pred_count.get(i, 1) + 1
        #print(pred_count, mask_count)
        #mask_count = collections.OrderedDict(sorted(mask_count.items(), key=lambda t: t[1], reverse=True))
        #pred_count = collections.OrderedDict(sorted(pred_count.items(), key=lambda t: t[1], reverse=True))

        #entries = 40
        #width = 0.4

        #pred_vals = []
        #for key in list(mask_count.keys())[:entries]:
        #   if key in pred_count.keys():
        #       pred_vals.append(pred_count[key])
        #   else:
        #       pred_vals.append(0)


        #plt.rc('ytick', labelsize=6) 
        #plt.barh(list(mask_count.keys())[:entries], list(mask_count.values())[:entries], -width, color='r', label="N times token masked", align='edge')
        #plt.barh(list(mask_count.keys())[:entries], pred_vals, width, color='b', label="N times token predicted", align='edge')
        #print(pred_vals)
        #plt.xlabel('Times')
        #plt.ylabel('Token')
        #plt.title('Token frequency among masks and predictions (Epoch ' + str(epoch) + ')')
        #plt.legend()
        #plt.show()
        #plt.savefig("barcharts/epoch_" + str(epoch) + "_" + str_code + ".pdf", format="pdf")
        #plt.cla()
        

 

    @staticmethod
    def check_top_x_mask_predictions(mask_predictions,mask_labels,x):
        return torch.sum(mask_predictions[:,:x] == mask_labels.unsqueeze(1).expand((mask_labels.shape[0],x))).item()


    def save(self, epoch, file_path):
        """
        Saving the current ELECTRA model on file_path

        :param epoch: current epoch number
        :param file_path: model output directory
        """
        output_file_path = file_path+"_epoch{}".format(epoch)
        if self.hardware == "parallel":
            #pdb.set_trace()
            self.electra.module.generator.save_pretrained(output_file_path+"_gen")
            torch.save(self.electra.module.embed_layer.state_dict(),output_file_path+"_gen_embed")
        else:
            #self.electra.generator.save_pretrained(output_file_path+"_gen")
            #torch.save(self.electra.generator.embed_layer.state_dict(),output_file_path+"_gen_embed")
            torch.save(self, output_file_path + "_gen.pth")
