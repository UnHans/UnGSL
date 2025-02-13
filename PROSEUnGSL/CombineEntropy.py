import torch
Entropy0=torch.load("/home/hs/PROSE/PROSE_pubmed/PROSEPubmedEntropy0"+".pt",map_location='cuda:0')
Entropy1=torch.load("/home/hs/PROSE/PROSE_pubmed/PROSEPubmedEntropy1"+".pt",map_location='cuda:0')
Entropy2=torch.load("/home/hs/PROSE/PROSE_pubmed/PROSEPubmedEntropy2"+".pt",map_location='cuda:0')
Entropy3=torch.load("/home/hs/PROSE/PROSE_pubmed/PROSEPubmedEntropy3"+".pt",map_location='cuda:0')
Entropy4=torch.load("/home/hs/PROSE/PROSE_pubmed/PROSEPubmedEntropy4"+".pt",map_location='cuda:0')
Entropy=(Entropy0+Entropy1+Entropy2+Entropy3+Entropy4)/5
torch.save(Entropy,"PROSEPubmedEntropy.pt")
a=torch.load("PROSEPubmedEntropy.pt")
print(a[:10],Entropy0[:10],Entropy1[:10],Entropy2[:10])