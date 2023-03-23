import torch 
training_args = torch.load('./test_training_args.bin')
print (training_args)

# from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
# device = "cuda:0"
# model = model.to(device)
# tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

# ARTICLE_TO_SUMMARIZE = (
# "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
# "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
# "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
# )
# inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
# print (inputs['input_ids'].shape)

# # Generate Summary
# import torch
# input_ids = torch.load('./input_ids.pt')
# decoder_input_ids = torch.load('./decoder_input_ids.pt')
# attention_mask = torch.load('./attention_mask.pt')
# print (input_ids.shape)
# print (decoder_input_ids.shape)
# print (attention_mask.shape)
# output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)