#ifndef __SEARCH_ENGINE_AWS__
#define __SEARCH_ENGINE_AWS__

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include "../searchEngine.h"

class SearchEngine_AWS : public SearchEngine
{
private:
    Aws::S3::S3Client s3_client;
public:
    SearchEngine_AWS();

    void setS3Client(Aws::S3::S3Client client) { s3_client = client; }
    
    // std::pair<std::string, std::string> getDocData(uint32_t docId) override;
    // std::pair<std::vector<Posting>, float> getWordPostings(const std::string& word) override;
};

#endif // __SEARCH_ENGINE_AWS__