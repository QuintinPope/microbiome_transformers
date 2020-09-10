import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader



from transformers import ElectraConfig,ElectraForPreTraining
import tqdm
import pdb

class ELECTRATrainer:
    """
    ELECTRATrainer make the pretrained ELECTRA model 
    """

    def __init__(self, electra: ElectraForPreTraining, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 100, log_file=None,training_checkpoint=None,input_embed=True):
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
        self.input_embed = input_embed
        self.sigmoid = torch.nn.Sigmoid()
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
        self.optim = SGD(self.electra.parameters(),lr=lr,momentum=0.9)

        self.log_freq = log_freq

        # clear log file
        if log_file:
            self.log_file = log_file
            if(training_checkpoint is None):
                with open(self.log_file,"w+") as f:
                    f.write("EPOCH,MODE,TOTAL CORRECT,AVG LOSS,TOTAL ELEMENTS,ACCURACY,MASK CORRECT,TOTAL MASK,MASK ACCURACY\n")
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

        cumulative_loss = 0.0

        total_correct = 0
        total_element = 0
        total_mask_correct = 0
        total_mask = 0
        for i, data in data_iter:
            #pdb.set_trace()
            #print(i)
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            #create attention mask
            #pdb.set_trace()
            zero_boolean = torch.eq(data["species_frequencies"],0)
            mask = torch.ones(zero_boolean.shape,dtype=torch.float).to(self.device)
            mask = mask.masked_fill(zero_boolean,0)

            
            # 1. forward the next_sentence_prediction and masked_lm model
            if self.input_embed:
                loss,scores = self.electra.forward(inputs_embeds=data["electra_input"],attention_mask=mask,labels=data["electra_label"])
            else:
                loss,scores = self.electra.forward(input_ids=data["electra_input"],attention_mask=mask,labels=data["electra_label"])                
            # 3. backward and optimization only in train
            if train:
                #self.optim_schedule.zero_grad()
                self.optim.zero_grad()
                if self.hardware == "parallel":
                    #pdb.set_trace()
                    loss.mean().backward()
                else:
                    loss.backward()
                #self.optim_schedule.step_and_update_lr()
                self.optim.step()
            #pdb.set_trace()
            scores = self.sigmoid(scores)
            predictions = torch.where(scores > 0.5,torch.tensor([1]).to(self.device),torch.tensor([0]).to(self.device))
            mask_predictions = torch.masked_select(predictions,data["mask_locations"])
            mask_token_labels = torch.masked_select(data["electra_label"],data["mask_locations"])
            total_mask_correct += torch.sum(mask_predictions == mask_token_labels).item()
            total_mask += mask_token_labels.shape[0]            
            
            #get accuracy for all tokens
            total_correct += torch.sum(predictions == data["electra_label"]).item()
            total_element += data["electra_input"].shape[0]*data["electra_input"].shape[1]

            log_loss = 0
            if self.hardware == "parallel":
                cumulative_loss += loss.sum().item()
                log_loss = loss.sum().item()

            else:
                cumulative_loss += loss.item()        
                log_loss = loss.item()    
            if i % self.log_freq == 0:
                data_iter.write("epoch: {}, iter: {}, avg loss: {},accuracy: {}/{}={:.2f}%, mask accuracy: {}/{}={:.2f}%, loss: {}".format(epoch,i,cumulative_loss/(i+1),total_correct,total_element,total_correct/total_element*100,total_mask_correct,total_mask,total_mask_correct/total_mask*100,log_loss))

 
            del data
            del mask
            del loss
            del scores
            del predictions
            del mask_predictions
            del mask_token_labels

        print("EP{}_{}, avg_loss={}, accuracy={:.2f}%".format(epoch,str_code,cumulative_loss / len(data_iter),total_mask_correct/total_mask*100))
        if self.log_file:
            with open(self.log_file,"a") as f:
                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,str_code,cumulative_loss/len(data_iter),total_correct,total_element,total_correct/total_element*100,total_mask_correct,total_mask,total_mask_correct/total_mask*100))

 
        
    def save(self, epoch, file_path):
        """
        Saving the current ELECTRA model on file_path

        :param epoch: current epoch number
        :param file_path: model output directory
        """
        output_file_path = file_path+"_epoch{}".format(epoch)
        if self.hardware == "parallel":
            self.electra.module.save_pretrained(output_file_path)
        else:
            self.electra.save_pretrained(output_file_path)