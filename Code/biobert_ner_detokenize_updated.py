import argparse




def detokenize(golden_path, pred_token_test_path, pred_label_test_path, output_dir, biotags_list, gold_standard):
    """convert word-piece bioBERT-NER results to original words (CoNLL eval format)

    Args:
        golden_path: path to golden dataset. ex) NCBI-disease/test.tsv
        pred_token_test_path: path to token_test.txt from output folder. ex) output/token_test.txt
        pred_label_test_path: path to label_test.txt from output folder. ex) output/label_test.txt
        output_dir: path to output result will write on. ex) output/
        gold_standard: boolean for gold standard in test.tsv (true) or not (false)

    Outs:
        NER_result_conll.txt
    """
    # read golden
    ans = dict()
    ans['toks'] = list()
    if gold_standard.lower() == 'true':
        ans['labels'] = list()
    total_tokens_no_sep = 0
    lineNoCount = 0
    print(golden_path) #test.tsv
    with open(golden_path, 'r') as in_:
        for line in in_:
            line = line.strip()
            if line == '':
                ans['toks'].append('[SEP]')
                lineNoCount += 1
                continue
            tmp = line.split()
            ans['toks'].append(tmp[0])
            total_tokens_no_sep += 1
            if gold_standard.lower() == 'true':
                ans['labels'].append(tmp[1])

    # len(ans['labels'])=ans['toks']-lineNoCount

    print('answer information:')
    print(len(ans['toks']), ans['toks'][:18])
    print('total tokens no sep', total_tokens_no_sep)
    if gold_standard.lower() == 'true':
        print(len(ans['labels']), ans['labels'][:18])
    print('line no count', lineNoCount)


    # read predicted
    pred = dict({'toks': [], 'labels': []})  # dictionary for predicted tokens and labels.
    #tokens predicted
    with open(pred_token_test_path, 'r') as in_:  # 'token_test.txt'
        for line in in_:
            line = line.strip()
            pred['toks'].append(line)

    #sep informaiton
    predicted_sep_count = 0
    #labels predicted
    with open(pred_label_test_path, 'r') as in_:  # 'label_test_3_epoch.txt'
        for line in in_:
            line = line.strip()
            if line in ['[CLS]', '[SEP]', 'X']:  # replace non-text tokens with O. This will not be evaluated.
                pred['labels'].append('O')

                if '[SEP]' == line:
                    predicted_sep_count += 1
                continue
            pred['labels'].append(line)

    print('pred information:')
    print(len(pred['toks']), pred['toks'][:18])
    print(len(pred['labels']), pred['labels'][:18])
    print('predicted sep count:', predicted_sep_count)

    if (len(pred['toks']) != len(pred['labels'])):  # Sanity check
        print("Error! : len(pred['toks']) != len(pred['labels']) : Please report us")
        raise

    bert_pred = dict({'toks': [], 'labels': []})
    for t, l in zip(pred['toks'], pred['labels']):
        if t in ['[CLS]', '[SEP]']:  # non-text tokens will not be evaluated.
            continue
        elif t[:2] == '##':  # if it is a piece of a word (broken by Word Piece tokenizer)
            bert_pred['toks'][-1] = bert_pred['toks'][-1] + t[2:]  # append pieces
        else:
            bert_pred['toks'].append(t)
            bert_pred['labels'].append(l)

    print('bert pred information:')
    print(len(bert_pred['toks']), bert_pred['toks'][:18])
    print(len(bert_pred['labels']), bert_pred['labels'][:18])
    # print(predicted_sep_count)


    if (len(bert_pred['toks']) != len(bert_pred['labels'])):  # Sanity check
        print("Error! : len(bert_pred['toks']) != len(bert_pred['labels']) : Please report us")
        raise


    ##fix the discrepancy between the answer tokenization and the bert_pred tokenization
    # ans['labels_updated'] = list()
    bert_pred['toks_updated'] = list()
    bert_pred['labels_updated'] = list()


    # if (len(ans['labels']) != len(bert_pred['labels'])): #don't always have the labels with predictions (no gold standard)
    if total_tokens_no_sep != len(bert_pred['labels']):
        ans_idx = 0 #index of token we are on
        bert_pred_tok_updated = ''
        bert_pred_labels_updated_list = []

        bert_pred_toks_split = []  # tokens
        bert_pred_lables_split = []  # labels

        exact_match_count = 0
        mismatch_count = 0
        unknown_count = 0
        sep_count = 0

        # print(len(bert_pred['toks']))
        # raise Exception('hold')
        for b, bert_pred_tok in enumerate(bert_pred['toks']):
            # print('original bert info looping:', b, bert_pred_tok)


            ##answer token information
            ans_tok = ans['toks'][ans_idx]
            # print('current_tokens:', bert_pred_tok, ans_tok)
            # print('bert pred token updated:', bert_pred_tok_updated)

            ##seems okay right now!! TODO: add if the ans token endswith the bert_pred_tok then take it finally! - check and finish up and move onto the next one - continue!!!

            ##collect the updated tokens
            if bert_pred_tok_updated and ans_tok == bert_pred_tok_updated:
                # print('collect the updated tokens!')
                # print('got here')

                # print(bert_pred_tok, bert_pred_tok_updated)
                # print(ans_tok)
                # print(ans_tok)
                # print(bert_pred_tok_updated)
                # print('labels, tokens')
                # print(len(bert_pred['labels_updated']), len(bert_pred['toks_updated']))

                bert_pred['toks_updated'] += [bert_pred_tok_updated]

                if len(list(set(bert_pred_labels_updated_list))) == 1:
                    bert_pred_labels_updated = list(set(bert_pred_labels_updated_list))[0]
                else:
                    for biotag in biotags_list:
                        if biotag in bert_pred_labels_updated_list:
                            bert_pred_labels_updated = biotag
                            break

                bert_pred['labels_updated'] += [bert_pred_labels_updated]
                # bert_pred_update = False
                bert_pred_tok_updated = ''  # reset!
                bert_pred_labels_updated_list = []
                #need to update the ans_tok with next one
                ans_idx += 1
                ans_tok = ans['toks'][ans_idx]
                # print(b)
                # print(ans_idx)
                # print(bert_pred_tok, ans_tok)

                # raise Exception('HOLD!')
            # else:
            #     pass

            ##split tokens
            elif bert_pred_toks_split:
                # print('split tokens')
                bert_pred['toks_updated'] += bert_pred_toks_split
                bert_pred['labels_updated'] += bert_pred_lables_split
                # need to update the ans_tok with next one
                ans_idx += 1
                ans_tok = ans['toks'][ans_idx]
                bert_pred_toks_split = []
                bert_pred_lables_split = []


            #skipping/missing stuff
            elif bert_pred_tok_updated and bert_pred_tok not in ans_tok:
                # print('skipping/missing stuff')
                #partial match
                if bert_pred_tok_updated and bert_pred_tok_updated in ans_tok:
                    if bert_pred_tok == '[UNK]':
                        pass

                    else:
                        # print('partial match')
                        # print('updated tok', bert_pred_tok_updated)
                        # print(bert_pred_tok, bert_pred['toks'][b-3:b+3])
                        # print(ans_tok, ans['toks'][ans_idx-3: ans_idx+5])
                        bert_pred['toks_updated'] += [ans_tok]

                        if len(list(set(bert_pred_labels_updated_list))) == 1:
                            bert_pred_labels_updated = list(set(bert_pred_labels_updated_list))[0]
                        else:
                            for biotag in biotags_list:
                                if biotag in bert_pred_labels_updated_list:
                                    bert_pred_labels_updated = biotag
                                    break

                        bert_pred['labels_updated'] += [bert_pred_labels_updated]
                        # bert_pred_update = False
                        bert_pred_tok_updated = ''  # reset!
                        bert_pred_labels_updated_list = []
                        ans_idx += 1
                        ans_tok = ans['toks'][ans_idx]


                #no bert_pred_tok_updated
                else:
                    pass
                    # print(bert_pred_tok, bert_pred_tok_updated)
                    # print(ans_tok)
                    # raise Exception('ERROR: PARTIAL MATCH PROBLEM!')



                ##reset all variable
                bert_pred_tok_updated = ''  # reset!
                bert_pred_labels_updated_list = []



            #TODO: fix skipping stuff - check if its the case - otherwise keep going - check in general! - changed 12.7.20

            if bert_pred_tok not in ans_tok or (bert_pred_tok_updated and bert_pred_tok_updated+bert_pred_tok not in ans_tok):
                # print('check stuff:', ans_tok)
                # print('skipping stuff!')
                ##made this higher range to check what is missing (found an example of 11)
                for j in range(ans_idx, ans_idx + 15):
                    ans_tok = ans['toks'][j]
                    # print('answer tok needing to find stuff:', ans_tok)
                    # skipped stuff - assume it is outside
                    # print('skipped stuff', ans_tok, bert_pred_tok)
                    if ans_tok == '[SEP]':
                        sep_count += 1
                        continue

                    ##unknown token
                    elif bert_pred_tok == '[UNK]':
                        # print('got here')
                        # bert_pred['toks_updated'] += [ans_tok]
                        # bert_pred['labels_updated'] += [bert_pred_label]
                        # # ans_idx += 1
                        # unknown_count += 1
                        # print('before ans tok', ans_tok)
                        ans_idx = j
                        # print('after ans tok:', ans['toks'][ans_idx])
                        # print('bert pred tok', bert_pred_tok)
                        # raise Exception('HOLD')
                        break

                    ##need to ensure that the match is one 3 total (one before the token, the token, and one after:
                    ##check if they are equal
                    elif ans_tok == bert_pred_tok:
                        ans_idx = j
                        # print('finished!')
                        # print(bert_pred['toks_updated'][-3:])
                        break
                    ##check if ans in bert pred tok (combined tokens in bert pred)
                    elif ans_tok in bert_pred_tok:
                        if ans_tok+ans['toks'][j+1] in bert_pred_tok:
                            ans_idx = j
                            # print('finished!')
                            # print(bert_pred['toks_updated'][-3:])
                            break
                        else:
                            # print('missing stuff 1', ans_tok)
                            bert_pred['toks_updated'] += [ans_tok]
                            bert_pred['labels_updated'] += ['O']  # assume it is Outside!
                    ##check if bert pred tok in ans (split up tokens in bert pred
                    elif bert_pred_tok in ans_tok:
                        if bert_pred_tok + bert_pred['toks'][b+1] in ans_tok:
                            ans_idx = j
                            # print('finished!')
                            # print(bert_pred['toks_updated'][-3:])
                            break
                        else:
                            # print('missing stuff 2', ans_tok)
                            bert_pred['toks_updated'] += [ans_tok]
                            bert_pred['labels_updated'] += ['O']  # assume it is Outside!
                    ##total miss of everything - no match at all
                    else:
                        # print('missing stuff 3', ans_tok)
                        bert_pred['toks_updated'] += [ans_tok]
                        bert_pred['labels_updated'] += ['O']  # assume it is Outside!




                    # elif ans_tok not in bert_pred_tok and bert_pred_tok not in ans_tok:
                    #     print('missing stuff', ans_tok)
                    #     bert_pred['toks_updated'] += [ans_tok]
                    #     bert_pred['labels_updated'] += ['O']  # assume it is Outside!
                    #     print(bert_pred['toks_updated'][-3:])
                    #     print(ans['toks'][ans_idx-2:ans_idx+2])
                    #     print(bert_pred['toks'][b-2:b+3])
                    #
                    # else:
                    #     ans_idx = j
                    #     print('finished!')
                    #     print(bert_pred['toks_updated'][-3:])
                    #     break

                    if j == ans_idx + 14:

                        print('bert pred token:', bert_pred_tok)
                        print(bert_pred['toks'][b-3:b+5])
                        print('bert pred token updated:', bert_pred_tok_updated)

                        print('answer token:', ans_tok)
                        print(ans['toks'][ans_idx-3:ans_idx+3])

                        print('full updated info:', bert_pred['toks_updated'][-15:])
                        print('ans info:', ans['toks'][ans_idx-5:ans_idx+15])
                        raise Exception('ERROR: DID NOT FIND THE MISSING PIECES CORRECTLY')

                # print(bert_pred['toks_updated'][-10:])

            ##answer token is SEP which we don't want so we skip
            if ans_tok == '[SEP]':
                ans_idx += 1
                ans_tok = ans['toks'][ans_idx]
                sep_count += 1
            else:
                pass

            bert_pred_label = bert_pred['labels'][b]

            ##exact match
            if ans_tok == bert_pred_tok:
                bert_pred['toks_updated'] += [bert_pred_tok]
                bert_pred['labels_updated'] += [bert_pred_label]
                ans_idx += 1
                exact_match_count += 1

                # ##reset our stuff!
                # if bert_pred_update:
                #
                #
                #     bert_pred_update = False

            ##unknown token
            elif bert_pred_tok == '[UNK]':

                # print('ans_tok for unknown info', ans['toks'][ans_idx-1], ans_tok)
                # print(bert_pred['toks'][b+1], bert_pred_tok)
                unknown_count += 1
                if bert_pred['toks'][b+1] in ans_tok:
                    # print('here!')
                    # print(ans_tok)
                    pass
                #     ans_idx = ans_idx - 1
                #     # ans_idx += 1

                elif bert_pred['toks'][b-1] != ans['toks'][ans_idx-1] and bert_pred['toks'][b-1] in ans['toks'][ans_idx-1]:
                    # print('UNKNOWN IS A PART OF THE SEPARATED TOKEN!')
                    # print(bert_pred_tok, ans_tok)
                    # print(bert_pred['toks'][b-1], ans['toks'][ans_idx-1])
                    # print(bert_pred['toks_updated'][-5:])
                    pass
                    # raise Exception('hold')

                else:
                    bert_pred['toks_updated'] += [ans_tok]
                    bert_pred['labels_updated'] += [bert_pred_label]
                    ans_idx += 1



                # print('here also')


            ##no match! - 3 scenarios
            else:
                #split up ans_tok - need to put bert_pred_tok back together
                if bert_pred_tok in ans_tok:
                    # print('split up ans_tok')
                    bert_pred_tok_updated += '%s' % (bert_pred_tok)
                    bert_pred_labels_updated_list += [bert_pred_label]
                    mismatch_count += 1

                    # print('answer info')
                    # print(ans_tok)
                    # print(ans_idx)
                    # print(ans['labels'][ans_idx - sep_count])

                    # print('bert info')
                    # print(bert_pred_tok)
                    # print(b)
                    # print(bert_pred_label)

                    ##update the bert_token
                    # bert_pred_update = True

                    # if bert_pred_update:
                    # bert_pred_tok_updated += '%s' %(bert_pred_tok)
                    # bert_pred_labels_updated_list += [bert_pred_label]
                    # mismatch_count += 1


                #split up bert_pred_token - need to separate bert_pred_tokens
                elif ans_tok in bert_pred_tok:
                    # print('split up bert pred tok')
                    ans_tok_updated = ''
                    bert_pred_toks_split = [] #tokens
                    bert_pred_lables_split = [] #labels
                    for i in range(ans_idx, ans_idx+10):
                        ans_tok_updated += '%s' %ans['toks'][i]
                        if ans_tok_updated == bert_pred_tok:
                            ans_idx = i #update the ans_idx to be the next one since we are done
                            if len(bert_pred_toks_split) != len(bert_pred_lables_split):
                                raise Exception('ERROR WITH LENGTHS OF SPLIT TOKENS AND LABELS!')
                            else:
                                break
                        elif ans_tok_updated in bert_pred_tok:
                            bert_pred_toks_split += [ans['toks'][i]]
                            bert_pred_lables_split += [bert_pred_label]

                        else:
                            print('error information:')
                            print(ans_tok)
                            print(bert_pred_tok)
                            print(bert_pred_tok_updated)
                            print(bert_pred_toks_split)
                            print(b)
                            print(bert_pred['toks'][b-5:b+10])
                            print(ans_idx)
                            print(ans['toks'][ans_idx-10:ans_idx+20])
                            print(ans['toks'][ans_idx])
                            print(bert_pred['toks_updated'][ans_idx-sep_count-10:])
                            # print('hello', bert_pred['toks_updated'][:ans_idx])
                            raise Exception('ERROR: CANNOT SPLIT UP THE BERT_PRED_TOKEN TO BE WHAT ANS_TOK WAS!')

                    # raise Exception('HOLD!')
                else:
                    pass




            # print('summary - ans then bert_pred')
            # print(len(ans['toks']), len(ans['labels']))
            # print(len(bert_pred['toks_updated']), len(bert_pred['labels_updated']))
            ##make sure we are adding well
            if len(bert_pred['labels_updated']) != ans_idx - sep_count:

                print(len(bert_pred['labels_updated']), ans_idx - sep_count)
                raise Exception('ERROR WITH ENSURING WE ADD AT THE SAME RATES BETWEEN BERT_PRED UPDATED LABELS AND ANS_IDX!')

            if len(bert_pred['labels_updated']) != len(bert_pred['toks_updated']):
                print('labels, tokens')
                print(len(bert_pred['labels_updated']), len(bert_pred['toks_updated']))
                raise Exception('ERROR WITH KEEPING BERT_PRED LABELS AND TOKENS THE SAME!')

            # if len(bert_pred['labels_updated']) > 54045:
            #     print('pred info')
            #     print(b, bert_pred_tok)
            #     print('ans info')
            #     print(ans_idx, ans_tok)
            #     print('labels, tokens')
            #     print(len(bert_pred['labels_updated']), len(bert_pred['toks_updated']))
            #     raise Exception('ERROR WITH TOO LARGE!')


        ##TODO: need final update just in case the updated stuff is at the end!
        if bert_pred_tok_updated:
            bert_pred['toks_updated'] += [bert_pred_tok_updated]
            bert_pred['labels_updated'] += [bert_pred_labels_updated]

    print('exact matches:', exact_match_count)
    print('mismatches', mismatch_count)
    print('unknowns', unknown_count)
    print('ans_index', ans_idx)

    #SANITY CHECKS
    #getting through all bert pred tokens
    if exact_match_count+mismatch_count+unknown_count != len(bert_pred['toks']):
        raise Exception('ERROR WITH CAPTURING ALL BERT PRED TOKENS!')
    else:
        pass

    if bert_pred_tok != bert_pred['toks'][-1]:
        raise Exception('ERROR WITH GETTING THROUGH ALL OF BERT_PRED_TOKENS by tokens!')
    else:
        pass

    #getting through all anser tokens
    if ans_idx != len(ans['toks'])-1 and ans_idx != len(ans['toks'])-2:
        print(ans_idx)
        print(len(ans['toks']))
        print(ans['toks'][-1])
        raise Exception('ERROR WITH ANS_IDX CAPTURING ALL OF THE ANSWER TOKENS!')
    else:
        pass




    # if (len(ans['labels']) != len(bert_pred['labels'])):  # Sanity check between the answer and prediction label files
    #     print(len(ans['labels']), len(bert_pred['labels']))
    #     print("Error! : len(ans['labels']) != len(bert_pred['labels']) : Please report us")
    #     raise
    
    # if (len(ans['labels']) != len(bert_pred['labels_updated'])):  # Sanity check between the answer and prediction label files
    if total_tokens_no_sep != len(bert_pred['labels_updated']):
        if gold_standard.lower() == 'true':
            print('answer labels len', len(ans['labels']))
        print('bert_pred labels updated length', len(bert_pred['labels_updated']))
        print('total tokens no sep', total_tokens_no_sep)
        print('ans info')
        print(ans_tok)
        print(ans['toks'][-4:])
        print('bert info')
        print(bert_pred_tok, bert_pred_tok_updated)

        print(bert_pred['toks_updated'][-4:])
        print("Error! : len(ans['labels']) != len(bert_pred['labels_updated']) : Please report us")
        raise Exception()

    # ##output the stuff together in conll format
    # with open(output_dir + '/NER_result_conll.txt', 'w') as out_:
    #     idx = 0
    #     for ans_t in ans['toks']:
    #         if ans_t == '[SEP]':
    #             out_.write("\n")
    #         else:
    #             out_.write(
    #                 "%s %s-MISC %s-MISC\n" % (bert_pred['toks'][idx], ans['labels'][idx], bert_pred['labels'][idx]))
    #             idx += 1

    ##output the stuff together in conll format
    print('final information:')
    print(len(ans['toks']))
    print(sep_count)
    print(len(ans['labels']))
    print(len(bert_pred['toks_updated']))
    print(len(bert_pred['labels_updated']))
    # raise Exception('Hold!')

    if gold_standard.lower() == 'true':
        with open(output_dir + '/NER_result_conll.txt', 'w') as out_:
            out_.write('%s\t%s\t%s\n' %('TOKEN', 'TRUE', 'PREDICTED'))
            idx = 0
            for ans_t in ans['toks']:
                if ans_t == '[SEP]':
                    out_.write("\n")
                else:
                    out_.write(
                        "%s\t%s\t%s\n" % (bert_pred['toks_updated'][idx], ans['labels'][idx], bert_pred['labels_updated'][idx]))
                    idx += 1


    # else:
    with open(output_dir + '/NER_predict_conll.txt', 'w') as out_:
        out_.write('%s\t%s\n' % ('TOKEN', 'PREDICTED'))
        idx = 0
        for ans_t in ans['toks']:
            if ans_t == '[SEP]':
                out_.write("\n")
            else:
                out_.write(
                    "%s\t%s\n" % (
                    bert_pred['toks_updated'][idx], bert_pred['labels_updated'][idx]))
                idx += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--token_test_path', type=str, help='')
    parser.add_argument('--label_test_path', type=str, help='')
    parser.add_argument('--answer_path', type=str, help='')
    parser.add_argument('--output_dir', type=str, help='')
    parser.add_argument('--biotags', type=str, help='')
    parser.add_argument('--gold_standard', type=str, help='', default='true')
    args = parser.parse_args()

    biotag_list = args.biotags.split(',')
    detokenize(args.answer_path, args.token_test_path, args.label_test_path, args.output_dir, biotag_list, args.gold_standard)
