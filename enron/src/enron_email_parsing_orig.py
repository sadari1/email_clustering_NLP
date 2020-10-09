# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

#%%

import os
import hashlib
import spacy
import numpy as np
import pandas as pd 
import time 
# %%

# Refers to another project folder with the full enron email data to save space.
enron_data_path = "../../../emailassistant/data/enron_mail_20150507/maildir"
# normal_path = "../../data/enron_mail_20150507/maildir"

dirs = [os.path.join(root, name)
for root, dirs, files in os.walk(enron_data_path)
for name in files]


# %%

# A list of preselected subjects to narrow down the focus of the entire dataset to just these subjects. 
subject_list = ["Western Wholesale Activities - Gas & Power Conf. Call Privileged & Confidential Communication", "Analyst Bryan Hull", "Do you still access data from Inteligence Press online??", "2nd lien info. and private lien info - The Stage Coach", "SM134 Proforma.xls", "West Gas 2001 Plan", "65th BD for Nea", "gas storage model", "FIMAT loan agreement", "mkts", "Commercials", "PG&E Energy Trading", "ng views + wager", "New number and address", "Mill Run report through May with Virtual Mode corrections.", "HR Position", "Sat night - the plan", "Interpreting Curves Data", "EnronOnline", "Forward-forward Vol Implementation Plan", "Houston Street", "trading with Campbell", "PG&E Energy Trading", "details for long term flat price swap on Nat Gas Houston Ship Channel Inside FERC", "RISK Magazine Interview", "technical help for interviewing traders"]

# %%

# Parsing code begins here.
spacySm = spacy.load("en_core_web_sm")

def parse_and_return(fields, key):

    for f in range(len(fields)):
        if key in fields[f]:
            if fields[f][len(key)+1:] == '':
                return "None"
            else:
                return fields[f][len(key)+1:]

    return "None"    

def get_hash(message):
    message = bytes(message, 'utf-8')
    txt = hashlib.sha1()
    txt.update(message)
    return txt.hexdigest()


# %%

# The parser that retrieves the desired fields given a list of email directories.
def emails_to_df(dirs):
    
    field_keys = ["Message-ID", "Date", "From", "To", "Subject", "Cc", "Bcc"]
    pd_array = []

    for dir in dirs:
        array = []
        
        
        reader = open(dir, 'r')
        content = str(reader.read())
        reader.close()

        found = False
        
        for f in subject_list:
            if f in content:
                found = True
        
        if not found:
            continue
        
        # Split content by new line
        sentences = content.split('\n')

        # Find where the message starts 
        fields = np.array([sent.split(':')[0] for sent in sentences])
        message_index = np.argwhere(fields == 'X-FileName')[0][0]

        # Body is right after where "X-FileName" is
        body = sentences[message_index+1:]
        message = ""

        for sent in body:
            message += sent + "\n"

        # Now to parse relevant email fields
        fields = sentences[:message_index+1]


        # message_id = parse_and_return(fields, "Message-ID:")[1:-1].split('.')
        # message_id = message_id[0] + '.' + message_id[1]

        field_dict = {}

       
        for f in field_keys:
            field_dict[f] = parse_and_return(fields, f"{f}:")
            array.append(field_dict[f])


        field_dict["Message-ID"] = field_dict["Message-ID"].split("<")[1][:-1]#field_dict["Message-ID"][1:-1].split('.')

        # Go fix the message ID to be correct
        array[0] = field_dict["Message-ID"]

        doc = spacySm(message)

        entities = []
        tokens = []
        tags = []

        for x in doc.ents:
            tokens.append(x.text)
            tags.append(x.label_)
        entities.append([tokens, tags])

        array.append(message)
        array.append(entities)
        pd_array.append(array)
        # print( pd.DataFrame(array))
    field_keys.append("Message")
    field_keys.append("Entity Vector")
    return pd.DataFrame(pd_array, columns=field_keys)


# %%

# Can filter the number of emails as desired. A timer logs job execution time.
tic = time.time()
a = emails_to_df(dirs[:])
toc = time.time()
print(f"Job took {toc-tic} seconds")
# a.head()

# %%

try:
    os.mkdir("../output")
except:
    _ = ""
    
# Output the data to a CSV. 
a.to_csv("../output/enron_from_to_largeset_2.csv", index=False)


