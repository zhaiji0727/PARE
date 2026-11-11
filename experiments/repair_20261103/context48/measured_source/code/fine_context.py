"""Fine context geometry; detector centers and suppression are frozen.

Only the crop support changes from half=12 to half=24. No cached scene,
coarse, aggregate, or image-level statistic is generated or relabelled here.
"""
import numpy as np
from saqre_phys import to_intensity, estimate_enl, ca_cfar, type_features


def select_scatterer_centers(g, det, max_sc=8, suppression_half=12):
    """Exact center-selection loop factored from frozen scatterer_patches."""
    if max_sc != 8 or suppression_half != 12:
        raise ValueError('This fixed study preserves max8 and suppression half12')
    H, W = g.shape
    ys, xs = np.where(det)
    if len(xs) == 0:
        return []
    values = g[ys, xs]
    eligible = np.ones(len(xs), dtype=bool)
    min_distance = np.full(len(xs), np.inf)
    center_distance = (xs - (W - 1) / 2.0) ** 2 + (ys - (H - 1) / 2.0) ** 2
    chosen = []
    while eligible.any() and len(chosen) < max_sc:
        ties = np.flatnonzero(eligible & (values == values[eligible].max()))
        index = ties[np.argmax(min_distance[ties]) if chosen else np.argmin(center_distance[ties])]
        x, y = int(xs[index]), int(ys[index])
        chosen.append((x, y))
        eligible &= (np.abs(xs - x) > suppression_half) | (np.abs(ys - y) > suppression_half)
        distance = (xs.astype(np.float64) - x) ** 2 + (ys.astype(np.float64) - y) ** 2
        min_distance = np.minimum(min_distance, distance)
    return chosen


def boxes_from_centers(centers, shape, crop_half):
    if crop_half not in (12, 24):
        raise ValueError('Only original24 verification and fixed context48 are allowed')
    H, W = shape
    boxes = []
    for x, y in centers:
        x0 = max(0, x - crop_half); y0 = max(0, y - crop_half)
        x1 = min(W, x + crop_half); y1 = min(H, y + crop_half)
        boxes.append((x0, y0, x1, y1))
    return boxes


def features_for_box(g, det, c_star, box):
    """Exact original feat_for arithmetic, using original full-image c_star."""
    x0, y0, x1, y1 = box
    sub = g[y0:y1, x0:x1]
    if sub.size == 0:
        sub = g
    mu = float(sub.mean())
    sig = float(sub.std())
    cv = sig / (mu + 1e-4)
    t = max(cv - c_star, 0.0)
    sub_det = det[y0:y1, x0:x1]
    cf = float(sub_det.mean()) if sub_det.size else 0.0
    tf = type_features(sub, cf)
    return {'mu': mu, 'cv': cv, 't': t, 'typefeat': tf}


def extract_fine_context(image, crop_half=24):
    image = image.convert('RGB')
    g = to_intensity(image)
    L = estimate_enl(g)
    c_star = 1.0 / np.sqrt(L)
    nu = float(np.quantile(g, 0.05))
    det = ca_cfar(g)
    centers = select_scatterer_centers(g, det)
    boxes = boxes_from_centers(centers, g.shape, crop_half)
    features = [features_for_box(g, det, c_star, box) for box in boxes]
    return {'image': image, 'g': g, 'det': det, 'centers': centers, 'boxes': boxes,
            'features': features, 'L': L, 'c_star': float(c_star), 'nu': nu,
            'crops': [image.crop(box) for box in boxes]}


def validate_overlay(base, overlay, base_cache_sha256, expected_source_sha256=None):
    import torch
    if overlay.get('schema') != 'pare.fine-context-overlay.v1' or overlay['ids'] != base['ids']:
        raise ValueError('Wrong overlay schema or image order')
    contract = overlay['contract']
    if contract['base_cache_sha256'] != base_cache_sha256 or contract['crop_half'] != 24 or contract['suppression_half'] != 12:
        raise ValueError('Overlay is not bound to the original cache/fixed geometry')
    if contract['max_fine_crops'] != 8 or overlay['D'] != base['D'] or base['D'] != 512 or base['Mf'] != 8:
        raise ValueError('Wrong capacity/dimension')
    if expected_source_sha256 is not None and contract['source_sha256'] != expected_source_sha256:
        raise ValueError('Overlay extraction sources changed')
    n = len(base['ids']);d=base['D']
    shapes = {'f_emb':(n,8,d),'f_mu':(n,8),'f_t':(n,8),'f_tf':(n,8,4),'f_n':(n,),
              'centers':(n,8,2),'boxes':(n,8,4),'image_sizes':(n,2)}
    for key,shape in shapes.items():
        expected_dtype = torch.float32 if key in ('f_emb','f_mu','f_t','f_tf') else torch.int64
        if overlay[key].dtype != expected_dtype or tuple(overlay[key].shape) != shape or not torch.isfinite(overlay[key]).all():
            raise ValueError(f'Invalid overlay field {key}')
    if not torch.equal(overlay['f_n'],base['f_n']):
        raise ValueError('Context change altered the frozen center count')
    mask=torch.arange(8)[None,:]<overlay['f_n'][:,None]
    for key in ('f_emb','f_mu','f_t','f_tf'):
        if torch.count_nonzero(overlay[key][~mask]):
            raise ValueError(f'Nonzero padded entries: {key}')
    if (overlay['image_sizes'] <= 0).any():
        raise ValueError('Invalid native image dimensions')
    geometry=[]
    for i,name in enumerate(base['ids']):
        count=int(overlay['f_n'][i]);shape=overlay['image_sizes'][i].tolist()
        centers=overlay['centers'][i,:count].tolist();boxes=overlay['boxes'][i,:count].tolist()
        if boxes != [list(b) for b in boxes_from_centers(centers,shape,24)]:
            raise ValueError('Overlay boxes do not match native clipped geometry')
        if any(not (0 <= x < shape[1] and 0 <= y < shape[0]) for x,y in centers):
            raise ValueError('Overlay center outside image')
        geometry.append({'id':name,'centers':centers,'boxes':boxes,'shape':shape})
    import hashlib,json
    digest=hashlib.sha256(json.dumps(geometry,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest != contract['geometry_sha256']:
        raise ValueError('Overlay center identities/order differ from full original-metadata validation')
    if mask.any() and not torch.allclose(overlay['f_emb'][mask].norm(dim=-1),torch.ones(int(mask.sum())),atol=1e-5,rtol=1e-5):
        raise ValueError('Fine embeddings are not normalized')


def compose_overlay(base, overlay, config, base_cache_sha256, expected_source_sha256=None):
    """In-memory composition only; explicitly does not claim old cache provenance."""
    if config['B'] != 0:
        raise ValueError('Original aggregate contains old fine crops; B must remain0')
    validate_overlay(base,overlay,base_cache_sha256,expected_source_sha256)
    composed = dict(base)
    for key in ('f_emb','f_mu','f_t','f_tf','f_n'):
        composed[key] = overlay[key]
    composed['provenance']={'schema':'pare.in-memory-base-plus-fine-overlay.v1',
                            'base_provenance':base['provenance'],'fine_overlay_contract':overlay['contract'],
                            'aggregate_is_original_and_must_not_be_used':True}
    return composed
