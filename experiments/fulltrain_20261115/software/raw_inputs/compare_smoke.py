"""CPU-only comparison of a new prefix cache with an explicit archived cache."""
import argparse
import json
from pathlib import Path
from input_contract import require, sha, write


def compare(new, reference):
    import torch
    expected = {'scene', 'agg', 'nu', 'L', 'c_star'} | {
        branch + '_' + field for branch in ('c', 'f') for field in ('emb', 'mu', 't', 'tf', 'n')}
    require({key for key, value in new.items() if torch.is_tensor(value)} == expected
            and {key for key, value in reference.items() if torch.is_tensor(value)} == expected,
            'Incomplete or extra cache tensor fields')
    require(len(new['ids']) == len(set(new['ids'])), 'Duplicate new IDs')
    require(new['D'] == reference['D'] and (new['Mc'], new['Mf']) == (reference['Mc'], reference['Mf']),
            'Cache dimensions differ')
    lookup = {name: i for i, name in enumerate(reference['ids'])}
    require(len(lookup) == len(reference['ids']), 'Duplicate reference IDs')
    require(set(new['ids']) <= set(lookup), 'New image IDs absent from reference')
    index = torch.tensor([lookup[name] for name in new['ids']])
    fields = {}
    for key, value in new.items():
        if not torch.is_tensor(value):
            continue
        require(key in reference and reference[key].dtype == value.dtype, 'Tensor schema differs: ' + key)
        old = reference[key][index]
        require(value.shape == old.shape, 'Tensor shape differs: ' + key)
        delta = (value.double() - old.double()).abs()
        fields[key] = {'exact': torch.equal(value, old),
                       'max_abs_difference': float(delta.max()) if delta.numel() else 0.0,
                       'unequal_elements': int((value != old).sum())}
    require(fields, 'No tensor fields')
    return {'all_tensor_fields_exact': all(v['exact'] for v in fields.values()), 'fields': fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-cache', type=Path, required=True)
    parser.add_argument('--reference-cache', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Refuse existing comparison report')
    require(sha(args.reference_cache) == args.reference_sha256, 'Reference file hash differs')
    before = {str(p): sha(p) for p in (args.new_cache, args.reference_cache)}
    import torch
    new, reference = [torch.load(p, map_location='cpu', weights_only=True)
                      for p in (args.new_cache, args.reference_cache)]
    result = compare(new, reference)
    require(before == {str(p): sha(p) for p in (args.new_cache, args.reference_cache)}, 'Inputs changed')
    result.update(schema='pare.portable-raw-smoke-alignment.v1', input_sha256=before,
                  image_ids=new['ids'], dimension=new['D'],
                  inference_performed=False, gpu_used=False,
                  scope='Cache tensors only. A tiny sample does not verify full raw reconstruction or text alignment.')
    write(args.out, result)
    print(json.dumps({'all_tensor_fields_exact': result['all_tensor_fields_exact'], 'out': str(args.out)}))


if __name__ == '__main__':
    main()
