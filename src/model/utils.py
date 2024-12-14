def format_result(predicted_answer, metrics):
    dict_result = dict()
    dict_result["answer"] = predicted_answer
    dict_result["bleu"] = metrics[0]
    dict_result["rouge1"] = metrics[1]
    dict_result["rouge2"] = metrics[2]
    dict_result["rougeL"] = metrics[3]
    dict_result["rougeLsum"] = metrics[4]
    dict_result["precision"] = metrics[5]
    dict_result["recall"] = metrics[6]
    dict_result["f1"] = metrics[7]
    return dict_result