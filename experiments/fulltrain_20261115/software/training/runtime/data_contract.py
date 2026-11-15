"""Pure data-index helpers; no automatic reads, tensors, model imports or execution."""
import csv
import hashlib
import json
from reference.old_retrieval_queries import select_queries


def require(condition,message):
    if not condition:raise ValueError(message)

def objsha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def textsha(value):return hashlib.sha256(value.encode('utf-8')).hexdigest()

def load_captions(path):
    # Same whitespace, column and insertion-order semantics as old train_saqre.py.
    caps={}
    with open(path) as stream:
        rows=csv.DictReader(stream);key_image=rows.fieldnames[0]
        key_caption=[k for k in rows.fieldnames if 'caption' in k.lower() and 'type' not in k.lower()][0]
        for row in rows:
            name,caption=row[key_image].strip(),row[key_caption].strip()
            if caption:caps.setdefault(name,[]).append(caption)
    return caps

def training_index(ids,captions,validate_official=True):
    require(len(ids)==len(set(ids)),'Duplicate training IDs')
    idrow={name:i for i,name in enumerate(ids)}
    raw=[(idrow[name],text) for name,rows in captions.items() if name in idrow for text in rows]
    texts=sorted({text for _,text in raw});textrow={text:i for i,text in enumerate(texts)}
    pairs=[(i,textrow[text]) for i,text in raw]
    if validate_official:
        require((len(ids),len(pairs),len({i for i,_ in pairs}),len(texts))==(9181,31945,9181,18461),'Full training population differs')
    return {'pairs':pairs,'texts':texts,'pair_order_sha256':objsha(pairs),
            'training_text_sha256':objsha([textsha(t) for t in texts]),'training_ids_sha256':objsha(ids),
            'descriptor_mean_rows':list(range(len(ids)))}

def query_sets(ids,captions,validate_official=True):
    first,fc=select_queries(captions,ids,'first-caption',0)
    allrows,ac=select_queries(captions,ids,'all-captions',0)
    extra=[r for r in allrows if r['caption_index_one_based']>1]
    if validate_official:require((len(ids),len(first),len(allrows),len(extra))==(3688,3688,6190,2502),'Official query population differs')
    require(not fc['truncated'] and not ac['truncated'],'No truncation')
    firstrow={r['query_id']:i for i,r in enumerate(first)};extrarow={r['query_id']:i for i,r in enumerate(extra)}
    require(set(firstrow).isdisjoint(extrarow),'First/extra overlap')
    mapping=[['first',firstrow[r['query_id']]] if r['query_id'] in firstrow else ['extra',extrarow[r['query_id']]] for r in allrows]
    return {'first':first,'extra':extra,'all':allrows,'all_caption_row_map':mapping,
            'first_text_sha256':objsha([r['caption_sha256'] for r in first]),
            'extra_text_sha256':objsha([r['caption_sha256'] for r in extra]),
            'all_text_sha256':objsha([r['caption_sha256'] for r in allrows]),
            'gallery_ids_sha256':objsha(ids),'coverage':{'first-caption':fc,'all-captions':ac}}

def validate_observed_text_schema(shared,extra,backbone,dimension):
    # Takes metadata-only schema dictionaries, never assumes extra vectors are in shared['test'].
    expected={'backbone','protocol_sha256','freeze_sha256','training_text_sha256','test_text_sha256','train','test'}
    require(set(shared['keys'])==expected and shared['metadata']['backbone']==backbone,'Shared text keys/identity')
    require(shared['tensors']=={'train':{'shape':[18461,dimension],'dtype':'torch.float32'},'test':{'shape':[3688,dimension],'dtype':'torch.float32'}},'Shared train/first-caption shapes')
    require(extra['type']=='bare_tensor' and extra['shape']==[2502,dimension] and extra['dtype']=='torch.float32','Separate extra-caption tensor shape')
