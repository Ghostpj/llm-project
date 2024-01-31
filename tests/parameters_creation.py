## Parameters file creation for the test

# Columns : [splitter, vectorstores, chunk_size, chunk_overlap]

splitters = ['recursive', 'token']
chunk_sizes = [10, 50, 100, 500, 1000]
chunk_overlaps = [0, 10, 50, 100, 500]
vector_stores = ['chroma', 'qdrant']

with open("test_parameters.csv", "w") as file :
    for s in splitters :
        for vs in vector_stores :
            for cs in chunk_sizes :
                for co in chunk_overlaps :
                    if co < cs :
                        file.write(s + ',' + vs + ',' + str(cs) + ',' + str(co) + '\n')
