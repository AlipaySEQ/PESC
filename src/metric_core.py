def get(t):
    if isinstance(t[0],int):
        return t[0]
    else:
        return t[0][0]
def sort(arr):
    n = len(arr) 
    for i in range(n):
 
        for j in range(0, n-i-1):
            
            if get(arr[j]) > get(arr[j+1]) :
                arr[j], arr[j+1] = arr[j+1], arr[j]

def sent_metric_detect(preds, targs):
    assert len(preds) == len(targs)
    tp, targ_p, pred_p, hit = 0, 0, 0, 0
    for pred ,targ in zip(preds, targs):
        pred = list(pred)
        targ = list(targ)
        sort(pred)
        sort(targ)
    
        if targ != []:
            targ_p += 1
        if pred != []:
            pred_p += 1
        if len(pred) == len(targ) and all(p[0] == t[0] for p, t in zip(pred, targ)):
            hit += 1
        if pred != [] and len(pred) == len(targ) and all(p[0] == t[0] for p, t in zip(pred, targ)):
            
            tp += 1

    acc = hit / len(targs)
    p = tp / pred_p
    r = tp / targ_p
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    
    results = {
        'sent-detect-acc': acc * 100,
        'sent-detect-p': p * 100,
        'sent-detect-r': r * 100,
        'sent-detect-f1': f1 * 100,
    }
    return results

def sent_metric_correct(preds, targs):
    assert len(preds) == len(targs)
    tp, targ_p, pred_p, hit = 0, 0, 0, 0
    for (idx,pred), targ in zip(enumerate(preds), targs):
        
        pred = list(pred)
        targ = list(targ)
        sort(pred)
        sort(targ)
        if targ != []:
            targ_p += 1
       
        if pred != []:
            pred_p += 1
        if pred == targ:
            hit += 1
        if pred != [] and pred == targ:
            tp += 1
            
    
    acc = hit / len(targs)
    p = tp / pred_p
    r = tp / targ_p
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    results = {
        'sent-correct-acc': acc * 100,
        'sent-correct-p': p * 100,
        'sent-correct-r': r * 100,
        'sent-correct-f1': f1 * 100,
    }
    return results

def char_metric_detect(preds, targs):
    assert len(preds) == len(targs)
    tp, fp, tn, fn = 0, 0, 0, 0
    for pred, targ in zip(preds, targs):
        pred = [idx for idx, _ in pred]
        targ = [idx for idx, _ in targ]
        tp += len(set(pred) & set(targ))  # 该纠，纠了
        fn += len(set(targ) - set(pred))  # 该纠，未纠
        fp += len(set(pred) - set(targ))  # 不该纠，纠了
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    results = {
             'char-detect-p': p * 100,
             'char-detect-r': r * 100,
             'char-detect-f1': f1 * 100,
         }
    return results


def char_metric_correct(preds, targs):
    assert len(preds) == len(targs)
    tp, fp, tn, fn = 0, 0, 0, 0
    for pred, targ in zip(preds, targs):
        pred_idx = [idx for idx, _ in pred]
        targ_idx = [idx for idx, _ in targ]
        tp += len(set(pred) & set(targ))# 该纠，纠了, 且纠对了
        fn += len(set(targ_idx) - set(pred_idx))  # 该纠，未纠
        fp += len(set(pred_idx) - set(targ_idx))  # 不该纠，纠了
        fp += len(set(targ) - set(pred))-len(set(targ_idx) - set(pred_idx)) #纠了，没纠对
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    results = {
             'char-correct-p': p * 100,
             'char-correct-r': r * 100,
             'char-correct-f1': f1 * 100,
         }
    return results



