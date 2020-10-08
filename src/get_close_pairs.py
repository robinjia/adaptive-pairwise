import faiss
import logging

def faiss_get_pairs(embeddings1, embeddings2, nlist, nprobe):
    assert(embeddings1.shape[0] <= embeddings2.shape[0])

    xq = embeddings1.astype('float32')
    xb = embeddings2.astype('float32')


    assert(xq.shape[1] == xb.shape[1])
    d = xq.shape[1]

    logging.info("Building index")
    quantizer = faiss.IndexFlatL2(d)  
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.nprobe = nprobe
    
    logging.info("Training index")
    assert(not index.is_trained)
    index.train(xb)
    assert(index.is_trained)

    logging.info("Adding to index")
    index.add(xb)

    logging.info("Searching index")
    D, I = index.search(xq, k=100)

    assert(I.shape[0] == xq.shape[0])

    pairs = []
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            pairs.append( (i, I[i,j]) )

    return pairs
