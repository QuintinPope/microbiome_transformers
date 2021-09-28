from torch.utils.data import Dataset,WeightedRandomSampler
import tqdm
import torch
import random
import pdb
import numpy as np

class ELECTRADataset(Dataset):
    def __init__(self, samples, embedding_path,labels, random_delete=0.0, random_insert=0.0, augment_probability = 1.0):
        self.embeddings = np.load(embedding_path)
        self.samples = samples
        self.labels = labels
        self.seq_len = self.samples.shape[1]+1
        #Initialize cls token vector values
        #pdb.set_trace()

        #take average of all embeddings
        #self.cls = np.average(self.embeddings,axis=0)
        self.cls = np.zeros(self.embeddings.shape[1])
        self.frequency_index = self.samples.shape[2] - 1 
        self.cls_frequency = 1

        self.random_delete = random_delete
        self.random_insert = random_insert
        self.augment_probability = augment_probability


        #initialize mask token vector values

        #find max and min ranges of values for every feature in embedding space
        #create random embedding
        self.embedding_mins = np.amin(self.embeddings,axis=0)
        self.embedding_maxes = np.amin(self.embeddings,axis=0)
        self.mask = self.generate_random_embedding()


        self.padding = np.zeros(self.embeddings.shape[1])
        #add cls, mask, and padding embeddings to vocab embeddings
        self.embeddings = np.concatenate((self.embeddings,np.expand_dims(self.mask,axis=0)))
        self.embeddings = np.concatenate((self.embeddings,np.expand_dims(self.cls,axis=0)))
        self.embeddings = np.concatenate((self.embeddings,np.expand_dims(self.padding,axis=0)))
        self.mask_index = self.lookup_embedding(self.mask)
        self.cls_index = self.lookup_embedding(self.cls)
        self.padding_index = self.cls_index +1
        

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        
        #pdb.set_trace()
        sample = self.samples[item]
        #print(sample[:, 0], sample.shape, item)
        sorted_indices = np.argsort(sample[:,1])
        sample = sample[sorted_indices][::-1]
        electra_label = self.labels[item]

        input_len = np.count_nonzero(sample[:, 0])
        augment_datapoint = random.random() > self.augment_probability
        if electra_label and self.random_delete > 0 and augment_datapoint:
            num_to_delete = int(input_len * self.random_delete)
            delete_indexes = np.random.permutation(np.arange(input_len))[:num_to_delete]
            for i in delete_indexes:
                sample[i][0] = 0
                sample[i][1] = 0

        sorted_indices = np.argsort(sample[:,1])
        sample = sample[sorted_indices][::-1]

        if electra_label and self.random_insert > 0 and augment_datapoint:
            extra_index = np.random.randint(1, self.samples.shape[0] - 1)
            extra_label = self.labels[extra_index]
            while extra_index == item or not extra_label:
                extra_index = np.random.randint(1, self.samples.shape[0] - 1)
                extra_label = self.labels[extra_index]
            #print(extra_index, self.samples.shape)
            extra_sample = self.samples[extra_index]
            random.shuffle(extra_sample)

            #print(extra_sample[:, 0])
            extra_sample = extra_sample[extra_sample[:, 0] > 0]
            #print(extra_sample, extra_sample.shape)
            input_len = np.count_nonzero(sample[:, 0])
            tokens_to_sample = min(int(input_len * self.random_insert), 511 - input_len)
            #print(tokens_to_sample, sample.shape, input_len)
            offset = 0
            for i in range(min(tokens_to_sample, len(extra_sample))):
                potential_extra_token = extra_sample[i]
                if potential_extra_token[0] in sample[:, 0]:
                    offset += 1
                    continue
                sample[input_len + 1 + i - offset][0] = potential_extra_token[0]
                sample[input_len + 1 + i - offset][1] = potential_extra_token[1]
       
        sorted_indices = np.argsort(sample[:,1])
        sample = sample[sorted_indices][::-1] 
            
        cls_marker = np.array([[self.cls_index,self.cls_frequency]],dtype=np.float)
        sample = np.concatenate((cls_marker,sample))
        electra_input,frequencies = self.match_sample_to_embedding(sample)

        output = {"electra_input": torch.tensor(electra_input,dtype=torch.long),
                "electra_label": torch.tensor(electra_label,dtype=torch.long),
                "species_frequencies": torch.tensor(frequencies,dtype=torch.long),
                }

        return output

    def match_sample_to_embedding(self, sample):
        electra_input = sample[:,0].copy()
        frequencies = np.zeros(sample.shape[0])
        for i in range(sample.shape[0]):
            #pdb.set_trace()
            if sample[i,self.frequency_index] > 0:
                frequencies[i] = sample[i,self.frequency_index]
            else:
                electra_input[i] = self.padding_index
                

        return electra_input,frequencies

    def generate_random_frequency(self):
        return np.random.randint(self.frequency_min,self.frequency_max)

    def generate_random_embedding(self):
        return np.random.uniform(self.embedding_mins,self.embedding_maxes)

    def vocab_len(self):
        return self.embeddings.shape[0]

    def lookup_embedding(self,bug):
        return np.where(np.all(self.embeddings == bug,axis=1))[0][0]

#for creating weighted random sampler
def create_weighted_sampler(labels):
    labels_unique, counts = np.unique(labels,return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    #class_weights[1] = class_weights[1]/2
    example_weights = [class_weights[int(e)] for e in labels]
    #print("Example Weights:")
    #print(example_weights)
    sampler = WeightedRandomSampler(example_weights,len(labels))
    return sampler

#for using class weights in loss function
def create_class_weights(labels):
    labels_unique, counts = np.unique(labels,return_counts=True)
    class_weights = [1 / c for c in counts]
    print(class_weights)
    return class_weights
