"""Independent identities, count-based ranks and metrics; no model imports."""
import csv
import hashlib
import json
import math


def require(value,message):
    if not value:raise ValueError(message)


def objsha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def identities(gallery,captions_path,official=True):
    require(len(gallery)==len(set(gallery)),'Duplicate gallery identities')
    index={name:i for i,name in enumerate(gallery)};captions={}
    with open(captions_path) as stream:
        rows=csv.DictReader(stream);image_key=rows.fieldnames[0]
        caption_key=next(k for k in rows.fieldnames if 'caption' in k.lower() and 'type' not in k.lower())
        for row in rows:
            image,text=row[image_key].strip(),row[caption_key].strip()
            if text:captions.setdefault(image,[]).append(text)
    records=[];first=[]
    for image,texts in captions.items():
        if image not in index:continue
        for i,text in enumerate(texts,1):
            if i==1:first.append(len(records))
            records.append({'query_id':f'{image}#caption={i}','image_id':image,
                            'caption_sha256':hashlib.sha256(text.encode()).hexdigest(),
                            'gold':index[image]})
    if official:require((len(gallery),len(records),len(first))==(3688,6190,3688),'Official identity population differs')
    require(len(first)==len(gallery),'First captions must cover every image')
    return records,first


def count_ranks(scores,gold,chunk=128):
    """Exact float32 ties: 1 + greater count + equal earlier gallery count."""
    import torch
    require(scores.ndim==2 and scores.dtype==torch.float32 and scores.device.type=='cpu','Expected CPU float32 scores')
    require(len(gold)==len(scores) and len(scores)>0 and scores.shape[1]>0,'Score population differs')
    require(all(type(g) is int and 0<=g<scores.shape[1] for g in gold),'Invalid gold row')
    ranks=[];ties=[];columns=torch.arange(scores.shape[1])[None,:]
    for begin in range(0,len(scores),chunk):
        block=scores[begin:begin+chunk]
        require(torch.isfinite(block).all().item(),'Nonfinite score')
        target=torch.tensor(gold[begin:begin+chunk],dtype=torch.int64)
        positive=block[torch.arange(len(block)),target][:,None]
        equal=block==positive
        ranks.extend((1+(block>positive).sum(1)+(equal & (columns<target[:,None])).sum(1)).tolist())
        ties.extend(equal.sum(1).tolist())
    return ranks,ties


def metrics(ranks,gallery_size):
    if not ranks:return None
    require(all(type(r) is int and 1<=r<=gallery_size for r in ranks),'Invalid ranks')
    return {**{f'R@{k}':100*sum(r<=k for r in ranks)/len(ranks) for k in (1,5,10)},
            'MRR@10':100*math.fsum(1/r if r<=10 else 0 for r in ranks)/len(ranks)}


def compare_metrics(actual,expected):
    if expected is None:require(actual is None,'Empty group has reported metrics');return
    require(set(actual)==set(expected),'Metric keys differ')
    require(all(math.isfinite(actual[k]) and math.isclose(actual[k],v,rel_tol=0,abs_tol=1e-9)
                for k,v in expected.items()),'Reported metrics differ from independent ranks')


def summarize(scores,gold,records,first,grouping):
    ranks,ties=count_ranks(scores,gold)
    first_ranks=[ranks[i] for i in first]
    first_images=[records[i]['image_id'] for i in first]
    require(len(first_images)==len(set(first_images)),'Repeated first-caption image')
    by_image=dict(zip(first_images,first_ranks))
    groups=grouping['groups'];flat=[name for names in groups.values() for name in names]
    require(len(flat)==len(set(flat)) and set(flat)==set(by_image),'GSD groups do not partition all first-caption images')
    require(grouping['n_gallery_retained_for_all_subgroups']==scores.shape[1],'GSD gallery size differs')
    return {'ranks':ranks,'ties':ties,'first_ranks':first_ranks,
            'query_modes':{'first-caption':metrics(first_ranks,scores.shape[1]),
                           'all-captions':metrics(ranks,scores.shape[1])},
            'gsd':{name:{'n_queries':len(images),'metrics':metrics([by_image[i] for i in images],scores.shape[1])}
                   for name,images in groups.items()}}
