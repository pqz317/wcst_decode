# splitter that goes like: 
# every split, find blocks where rules match dimension, blocks where rules don't
# out of matching blocks, sample 1 block for every condition 
# (needs to ensure each condition occurs at least 2 times)
# sample 4 blocks from non-matching blocks
# so each time, construct 8 blocks for testing, rest of the blocks for training
#
# interested in a few test accuracies
# original acc, last n corrects, rule match dimension acc, rule doesn't match dimension acc. 

