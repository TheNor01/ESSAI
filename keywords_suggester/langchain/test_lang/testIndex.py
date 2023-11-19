#index 

"""

When indexing content, hashes are computed for each document, and the following information is stored in the record manager:

the document hash (hash of both page content and metadata)
write time
the source id -- each document should include information in its metadata to allow us to determine the ultimate source of this document

"""