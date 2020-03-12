def groupSequence(lst): 
    res = [[lst[0]]] 
  
    for i in range(1, len(lst)): 
        if lst[i-1]+1 == lst[i]: 
            res[-1].append(lst[i]) 
        else: 
            res.append([lst[i]]) 
    return res

def get_invmatchseq(lst_of_lst_invnumbers):
    output = []
    flag = 0
    for ind_i, i in enumerate(lst_of_lst_invnumbers):
        if ind_i+1 >= len(lst_of_lst_invnumbers):
            if len(lst_of_lst_invnumbers) == 1:
                return [(ind_i,ind_i)]
            break
        
        for ind_j, j in enumerate(lst_of_lst_invnumbers[ind_i+1:]):
            if any([True for x in i if x in j]):
                output.append((ind_i,ind_i+1+ind_j))
                flag = 1
    if flag == 0:
        flag = 0
        return [(ind_i,ind_i)]
    
    return output

def get_invmatchseq_order(inv_matched_seq,page_size):
    out = []
    tmp_dict = dict()
    for ind,i in enumerate(inv_matched_seq):
        frst = i[0]
        scnd = i[1]
        tmp = []
        for x in inv_matched_seq[ind+1:]:
            if frst in x or scnd in x:
                tmp.extend([i[0],i[1],x[0],x[1]])
        
        if len(tmp) == 0:
            tmp.extend([i[0],i[1]])
        
        flag = 0
        for i in tmp:
            if not tmp_dict.get(i,None):
                tmp_dict[i] = 1
                flag = 1

        if flag == 1:
            for i in tmp:
                for j in out:
                    if i in j:
                        tmp.remove(i)
            out.append([i for i in range(min(set(tmp)),max(set(tmp))+1)])
    
    new_out = []
    for i in range(0,page_size):
        if not any([True for x in out if i in x]):
            new_out.append(list(set([i for x in out if i not in x])))

    output = []
    for i in out+new_out:
        tmp_var = []
        for j in i:
            tmp_var.append(j+1)
        output.append(tmp_var)
    
    return output

def merge_groupedseq_invs(N):
    t_dict = dict()
    counter = 1
    flag = 0
    for ind, i in enumerate(N):
        for key,val in t_dict.items():
            if len(set(i).intersection(set(val))):
                t_dict[key] = val+i
                flag = 1

        if flag == 1:
            flag = 0
            continue

        tmp_lst = i
        for j in N[ind+1:]:
            if len(set(i).intersection(set(j))):
                tmp_lst.extend(j)

        t_dict[counter] = tmp_lst
        counter += 1
    t_dict = {key:list(set(val)) for key,val in t_dict.items()}
    
    return [t_dict[key] for key in sorted(t_dict)]

def get_index_ofinvvisualcues(ds):
    ds1 = len([item for sublist in ds for item in sublist]) #page_size
    ds2 = [i for i in range(1,ds1+1)]
    counter = 0
    tmp_lst = []
    for i in ds:
        tmp_lst.append(ds2[counter:counter+len(i)])
        counter += len(i)
    
    return tmp_lst
