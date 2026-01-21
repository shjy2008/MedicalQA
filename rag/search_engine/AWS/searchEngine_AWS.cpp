#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <cstdint>
#include <chrono>
#include "searchEngine_AWS.h"

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>

const std::string aws_s3_bucketName = "search-engine-pubmed-abstract";

SearchEngine_AWS::SearchEngine_AWS() {
	indexPath = "../";
	// indexPath = "/projects/sciences/computing/sheju347/MedicalQA/rag/search_engine/";
}

std::pair<std::string, std::string> SearchEngine_AWS::getDocData(uint32_t docId) {
	DocumentOffset offsetData = this->docOffsetTable[docId - 1];

	// AWS read index file stored in S3
	Aws::S3::Model::GetObjectRequest object_request;
	object_request.SetBucket(aws_s3_bucketName.c_str());
	object_request.SetKey("index_documents.bin");

	std::streamoff start = offsetData.offset;
	std::streamoff end = offsetData.offset + offsetData.docNoLength + offsetData.documentLength;
	object_request.SetRange("bytes=" + std::to_string(start) + "-" + std::to_string(end));

	auto get_object_outcome = s3_client.GetObject(object_request);
	if (!get_object_outcome.IsSuccess()) {
            std::cerr << "Failed to read S3 object: "
                      << get_object_outcome.GetError().GetMessage() << std::endl;
			return std::pair<std::string, std::string>("", "");
    }

	Aws::IOStream& s3_stream = get_object_outcome.GetResultWithOwnership().GetBody();

	// Read docNo
	std::string docNo(offsetData.docNoLength, '\0');
	s3_stream.read(&docNo[0], offsetData.docNoLength);

	// Read document
	std::string document(offsetData.documentLength, '\0');
	s3_stream.read(&document[0], offsetData.documentLength);

	// ---- Below is for direct access local index file ---
	// std::ifstream documentsFile;
	// documentsFile.open(indexPath + "index_documents.bin");
	// documentsFile.seekg(offsetData.offset, std::ifstream::beg);

	// // Read docNo
	// char* buffer = new char[offsetData.docNoLength];
	// documentsFile.read(buffer, offsetData.docNoLength);
	// std::string docNo(buffer, offsetData.docNoLength);
	// delete[] buffer;

	// // Read document
	// buffer = new char[offsetData.documentLength];
	// documentsFile.read(buffer, offsetData.documentLength);
	// std::string document(buffer, offsetData.documentLength);
	// delete[] buffer;

	return std::pair<std::string, std::string>(docNo, document);
}

std::pair<std::vector<Posting>, float> SearchEngine_AWS::getWordPostings(const std::string& word) {

	std::vector<Posting> postings;

	std::unordered_map<std::string, WordData>::iterator wordDataItr = this->wordToWordData.find(word);
	if (wordDataItr == this->wordToWordData.end()) {
		return std::pair<std::vector<Posting>, float>(postings, 0); // Can't find the word, return empty vector
	}

	WordData wordData = wordDataItr->second;
	uint32_t pos = wordData.postingsPos;
	uint32_t docCount = wordData.postingsDocCount;
	float impactScore = wordData.impactScore;

	// AWS read index file stored in S3
	std::streamoff offset = sizeof(uint32_t) * pos * 2;
	std::streamoff bytesToRead = docCount * sizeof(Posting);

	Aws::S3::Model::GetObjectRequest object_request;
	object_request.SetBucket(aws_s3_bucketName.c_str());
	object_request.SetKey("index_postings.bin");
	object_request.SetRange("bytes=" + std::to_string(offset) + "-" + std::to_string(offset + bytesToRead - 1));

	auto get_object_outcome = s3_client.GetObject(object_request);

    if (!get_object_outcome.IsSuccess()) {
        std::cerr << "Failed to read S3 object: "
                  << get_object_outcome.GetError().GetMessage() << std::endl;
        return std::pair<std::vector<Posting>, float>(postings, 0); // failed to read S3, return empty vector
    }

	Aws::IOStream& s3_stream = get_object_outcome.GetResultWithOwnership().GetBody();

	postings.resize(docCount);
	s3_stream.read(reinterpret_cast<char*>(postings.data()), bytesToRead);

	// ---- Below is for direct access local index file ---
	// // Seek and read wordPostings.bin to find the postings(docId and tf) of this word
	// wordPostingsFile.seekg(sizeof(uint32_t) * pos * 2, std::ifstream::beg); // * 2 because every doc has docId and term frequency

	// // Optimization: batch reading instead of reading docId and tf one by one
	// postings.resize(docCount);
	// wordPostingsFile.read(reinterpret_cast<char*>(postings.data()), docCount * sizeof(Posting));

	// for (uint32_t i = 0; i < docCount; ++i) {
	// 	uint32_t docId = 0;
	// 	uint32_t tf = 0;
	// 	wordPostingsFile.read((char*)&docId, 4);
	// 	wordPostingsFile.read((char*)&tf, 4);

	// 	postings.push_back(Posting(docId, tf));
	// }

	return std::pair<std::vector<Posting>, float>(postings, impactScore);
}
