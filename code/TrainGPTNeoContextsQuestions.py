#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip install transformers datasets')
import datasets
from datasets import load_dataset
import os
import time
import datetime
#from google.colab import drive
import pandas as pd
import seaborn as sns
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPTNeoForCausalLM, GPTNeoConfig, TextDataset
from transformers import AdamW, get_linear_schedule_with_warmup
import transformers
import nltk
nltk.download('punkt')
dataset = load_dataset("squad_v2")
dataset
# prepare and load dataset
df = pd.concat([pd.DataFrame(dataset['train']),pd.DataFrame(dataset['validation'])],axis=0)
# # of stratified context bootstrap samples, i.e. # of times each context is 'repeated'
samples = 5
max_length = 256
# Set the seed value all over the place to make this reproducible.
seed_val = 42
# some parameters I cooked up that work reasonably well
batch_size = 24
epochs = 5
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8
# this produces sample output every 100 steps
sample_every = 250
df_samples_i = []
for s in np.array(range(0,samples,1)):
  df_samples = df.groupby('context', group_keys=False).apply(lambda x: x.sample(1))
  df_samples = df_samples.sort_values(by='context')
  df_samples_i.append(df_samples.index)
  
  #df_samples = pd.concat([df_samples,df.groupby('context', group_keys=False).apply(lambda x: x.sample(1)])
  #df.groupby('context', group_keys=False).apply(lambda x: x.sample(1))
unique_df_sample_i = np.unique(np.array(df_samples_i).ravel())
print(len(unique_df_sample_i))
#samples = np.random.choice(unique_df_sample_i, 500, replace=False)
#subset = df.loc[samples]
sample_size = 10000
subset = df.loc[unique_df_sample_i].sort_values(by='context')[['context','question','answer']].head(sample_size)
cq = 'Context: ' + subset['context'] + '\nQuestion: ' + subset['question'] + '\n'
get_ipython().system('nvidia-smi')
doc_lengths = []
for cq_ in cq:
    # get rough token count distribution
    tokens = nltk.word_tokenize(cq_)
    doc_lengths.append(len(tokens))
doc_lengths = np.array(doc_lengths)
sns.distplot(doc_lengths)
# the max token length   
len(doc_lengths[doc_lengths > max_length])/len(doc_lengths)
np.average(doc_lengths)
reduced_qc = cq[doc_lengths < max_length]
# This is an example list of prompts, which will be tokenized
prompts = ["This is the first prompt.",
           "This is the second prompt."]
prompts = cq
# Load the GPT tokenizer.
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>',max_len=max_length) #gpt2-medium
print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(tokenizer.model_max_length))
print("The beginning of sequence token {} token has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))
class GPT2Dataset(Dataset):
    def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx] 
"""
dataset = TextDataset(cq, tokenizer, block_size=768)
# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))
dataset = GPT2Dataset(reduced_qc, tokenizer, max_length=256)
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
# I'm not really doing anything with the config buheret
#configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
configuration = GPTNeoConfig.from_pretrained('EleutherAI/gpt-neo-125M', output_hidden_states=False)
# instantiate the model
#model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M", config=configuration)
# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))
# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
#device = torch.device("cpu")
model.cuda()
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )
# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)
print(total_steps)
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))
import gc
how_many_gpus = torch.cuda.device_count()
for _ in range(how_many_gpus):
  torch.cuda.set_device(_)
  torch.cuda.empty_cache()
  gc.collect()
tokenizer.vocab_size
prompt = "<|startoftext|>Context:"
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
generated = generated.to(device)
print(generated)
total_t0 = time.time()
training_stats = []
model = model.to(device)
for epoch_i in range(0, epochs):
    # ========================================
    #               Training
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        model.zero_grad()        
        outputs = model(  b_input_ids,
                          labels=b_labels, 
                          attention_mask = b_masks,
                          token_type_ids=None
                        )
        loss = outputs[0]  
        batch_loss = loss.item()
        total_train_loss += batch_loss
        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))
            model.eval()
            sample_outputs = model.generate(
                                    #bos_token_id=random.randint(1,tokenizer.vocab_size),
                                    generated,
                                    #prompt = "<|startoftext|>Context: "
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = 256,
                                    top_p=0.95, 
                                    num_return_sequences=1
                                )
            for i, sample_output in enumerate(sample_outputs):
                  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            
            model.train()
        loss.backward()
        optimizer.step()
        scheduler.step()
    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
        
    #               Validation
    print("Running Validation...")
    model.eval()
    total_eval_loss = 0
    nb_eval_steps = 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        with torch.no_grad():        
            outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                            labels=b_labels)
          
            loss = outputs[0]  
        total_eval_loss += batch_loss        
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
# Display floats with two decimal places.
pd.set_option('precision', 2)
# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)
# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')
# A hack to force the column headers to wrap.
#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
# Display the table.
df_stats
# Use plot styling from seaborn.
sns.set(style='darkgrid')
# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])
plt.show()
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())
print('The GPT-2 model has {:} different named parameters.\n'.format(len(params)))
print('==== Embedding Layer ====\n')
for p in params[0:2]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== First Transformer ====\n')
for p in params[2:14]:
print('\n==== Output Layer ====\n')
for p in params[-2:]:
# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
output_dir = './model_save/'
# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Saving model to %s" % output_dir)
# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
import zipfile
zfName = './model_save/*'
foo = zipfile.ZipFile(zfName, 'w')
# Adding files from directory 'files'
#for root, dirs, files in os.walk('model-neo.bin'):
    #for f in files:
        #foo.write(os.path.join(root, f))
foo.write("./model_save/*")
import shutil
shutil.make_archive("model", "zip", "model_save")
ls /content/model_save
from google.colab import files
files.download('/content/model.zip')
get_ipython().system('ls -l --block-size=K ./model_save/')
get_ipython().system('ls -l --block-size=M ./model_save/pytorch_model.bin')
# Copy the model files to a directory in your Google Drive.
get_ipython().system('cp -r ./model_save/ $data_dir')
# # Load a trained model and vocabulary that you have fine-tuned
#model = GPT2LMHeadModel.from_pretrained(output_dir)
#tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
#model.to(device)
tokenizer.decode(32)
model.eval()
sample_outputs = model.generate(
                                generated, 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True, 
                                num_beams=3,
                                no_repeat_ngram_size=2,
                                top_k=50, 
                                max_length = 256,
                                top_p=0.95, 
                                num_return_sequences=3
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))