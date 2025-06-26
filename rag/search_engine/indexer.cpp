#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>

// Extract words from a text string
std::vector<std::string> extractWords(const std::string& text) {
	std::vector<std::string> words;

	std::string word;
	for (size_t i = 0; i < text.size(); ++i){
		if (std::isalpha(text[i]))
			word += std::tolower(text[i]);
		else if (std::isdigit(text[i]))
			word += text[i];
		// else if (text[i] == '-') { // Some words have '-', such as "well-being"
		// 	if (i > 0 && std::isalnum(text[i - 1])) { // Ignore words starting with '-'
		// 		word += text[i];
		// 	}
		// }
		else {
			if (word.length() > 0) {
				if (word.length() > 255) { // Length is stored in uint8_t, so truncate the word if its length > 255
					word = word.substr(0, 255);
				}
				words.push_back(word);
			}
			word = "";
		}
	}
	if (word.length() > 0)
		words.push_back(word);

	return words;
}

// Strip the spaces from the beginning and the end of a string
std::string stripString(const std::string& text) {
	if (text.length() == 0) {
		return "";
	}
	uint32_t start = 0;
	uint32_t end = text.length() - 1;

	while (start <= end && isspace(text[start])) {
		++start;
	}

	while (start <= end && isspace(text[end])) {
		--end;
	}
	return text.substr(start, end - start + 1);
}

class Indexer {

private:
	std::string fileName;

	// word -> [(docid_1, term frequency), (docid_1, term frequency), ...]
	// (docid: 1, 2, 3, ...)
	// e.g. {"aircraft": [(6, 1), ...], "first": [(5, 1), (6, 2), ...], ...}
	std::unordered_map<std::string, std::vector<std::pair<uint32_t, uint32_t> > > wordToPostings; 

	// Document length list
	// e.g. [159, 64, 48, 30, 106, 129, ...]
	std::vector<uint32_t> documentLengthList;

	// Save postings in batch, not create all postings before saving (memory issue)
	const uint32_t postingsBatchSize = 10; // Postings batch size. Save postings index every batch, then merge them
	uint32_t postingsBatchCounter = 0; // reset to 0 after each batch
	uint32_t postingsBatchIndex = 0; // increment after each batch

public:
	Indexer(std::string fileName) {
		this->fileName = fileName;
	}

	void saveIndexToFiles() {
		// Save document length list
		std::ofstream docLengthsFile("index_docLengths.bin"); // an uint32_t(4 byte) for each document length
		for (size_t i = 0; i < documentLengthList.size(); ++i) {
			docLengthsFile.write((const char*)&documentLengthList[i], 4);
		}
	}

	void savePostingBatch() {
		uint32_t wordCount = (uint32_t)this->wordToPostings.size();
		if (wordCount == 0)
			return;

		uint32_t batchIndex = this->postingsBatchIndex;
		++this->postingsBatchIndex;

		// Stored as (docId1 for word1, term frequency 1 for word1, docId2 for word1, tf2 for word1, 
		// 				docId1 for word2, tf1 for word2, ...) each in 4 bytes uint32_t
		std::ofstream postingsBatchFile("index_postings_" + std::to_string(batchIndex) + ".bin");
		
		// Stored as: 4 byte word count + [(wordLength(1 byte), word, pos(4 bytes), docCount(4 bytes)), ...]
		// -- pos: how many documents before the word's first document
		// -- docCount: how many documents the word appears in (vector's size) 
		std::ofstream wordsBatchFile("index_words_" + std::to_string(batchIndex) + ".bin");

		wordsBatchFile.write((const char*)&wordCount, 4); // 4 byte word count

		// Convert to vector and sort term->[<docId, tf>, ...] in alphabetical order
		std::vector<std::pair<std::string, std::vector<std::pair<uint32_t, uint32_t> > > > sortedPostings(
			this->wordToPostings.begin(), this->wordToPostings.end());
		std::sort(sortedPostings.begin(), sortedPostings.end(), 
				[](const auto& a, const auto& b) {
					return a.first < b.first;
				});
		
		uint32_t docCounter = 0;
		for (auto it = sortedPostings.begin(); it != sortedPostings.end(); ++it) 
		{
			std::string word = it->first;
			std::vector<std::pair<uint32_t, uint32_t> > postings = it->second;
			uint32_t docCount = postings.size();

			uint8_t wordLength = (uint8_t)word.length();
			wordsBatchFile.write((const char*)&wordLength, 1);
			wordsBatchFile.write(word.c_str(), wordLength);
			wordsBatchFile.write((const char*)&docCounter, 4);
			wordsBatchFile.write((const char*)&docCount, 4);

			for (uint32_t i = 0; i < docCount; ++i) {
				uint32_t docId = postings[i].first;
				uint32_t tf = postings[i].second;
				postingsBatchFile.write((const char*)&docId, 4);
				postingsBatchFile.write((const char*)&tf, 4);

				++docCounter;
			}
		}

		this->wordToPostings.clear();
	}

	void mergePostingsBatch() {
		// const uint32_t mergeFileCount = 3; // How many postings_xx.bin files to merge
		// const uint32_t mergeCount = std::ceil((float)this->postingsBatchIndex / mergeFileCount); // Merge how many times
		const uint32_t mergeFileCount = this->postingsBatchIndex;

		// [ batch1:[ <term, <pos, docCount> > , ...], batch2: [...] ]
		std::vector<std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t> > > > allBatchWordsData;

		// Read all batched words file: index_words_xx.bin
		for (uint32_t i = 0; i < mergeFileCount; ++i) {
			std::ifstream wordsBatchFile("index_words_" + std::to_string(i) + ".bin");
			
			// Get how many bytes the words.bin have
			wordsBatchFile.seekg(0, std::ifstream::end);
			uint64_t fileSize = wordsBatchFile.tellg();
			wordsBatchFile.seekg(0, std::ifstream::beg);

			// Batch reading is faster than reading byte by byte
			char* buffer = new char[fileSize];
			wordsBatchFile.read(buffer, fileSize);
			
			wordsBatchFile.close();

			char* pointer = buffer;

			uint32_t wordCount = *reinterpret_cast<uint32_t*>(pointer);
			pointer += 4;
			
			std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t> > > batchWordsData;

			for (uint32_t j = 0; j < wordCount; ++j) {
				uint8_t wordLength = *pointer; //*reinterpret_cast<uint8_t*>(pointer);
				++pointer;

				std::string word(pointer, wordLength);
				pointer += wordLength;

				uint32_t pos = *reinterpret_cast<uint32_t*>(pointer);
				pointer += 4;

				uint32_t docCount = *reinterpret_cast<uint32_t*>(pointer);
				pointer += 4;

				batchWordsData.push_back(std::pair<std::string, std::pair<uint32_t, uint32_t> >(word, std::pair<uint32_t, uint32_t>(pos, docCount)));
			}

			allBatchWordsData.push_back(batchWordsData);
		}

		std::unordered_map<uint32_t, uint32_t> batchIndexToWordIndex;
		std::vector<std::ifstream> postingsFileList(mergeFileCount);

		for (uint32_t i = 0; i < mergeFileCount; ++i) {
			postingsFileList[i].open("index_postings_" + std::to_string(i) + ".bin");
			batchIndexToWordIndex[i] = 0;
		}

		// Merge all batches into a single postings file and a single words file
		std::ofstream mergedPostingsFile("index_postings.bin");
		std::ofstream mergedWordsFile("index_words.bin");

		uint32_t docCounter = 0;
		std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t> > > allWordsData;

		while (true)
		{
			// Check if all words are processed
			bool finished = true;
			for (uint32_t i = 0; i < mergeFileCount; ++i) {
				uint32_t wordCount = allBatchWordsData[i].size();
				if (batchIndexToWordIndex[i] < wordCount) {
					finished = false;
					break;
				}
			}
			if (finished)
				break;
			
			// Merge all batch postings, one word at a time, in alphabetical order
			std::string currentWord = "";
			uint32_t currentWordDocCounter = 0;
			std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t> > batchIndexToPosAndDocCount;
			for (uint32_t i = 0; i < mergeFileCount; ++i) {
				uint32_t wordIndex = batchIndexToWordIndex[i];
				if (wordIndex >= allBatchWordsData[i].size()) { // If word list of a batch file is all done, don't need to process it
					continue;
				}
				std::pair<std::string, std::pair<uint32_t, uint32_t> >& currentWordData = allBatchWordsData[i][wordIndex];
				if (currentWord == "" || currentWordData.first <= currentWord) {
					if (currentWordData.first < currentWord) {
						batchIndexToPosAndDocCount.clear();
					}
					batchIndexToPosAndDocCount[i] = currentWordData.second;
					currentWord = currentWordData.first;
				}
			}

			int pos = docCounter; // How many docs before

			// {batch_index1: [<docId1, tf1>, ...], batch_index2: [<docId, tf, ...], ...}
			std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t> > > batchIndexToPostings; 
			for (auto itr = batchIndexToPosAndDocCount.begin(); itr != batchIndexToPosAndDocCount.end(); ++itr) {
				uint32_t batchIndex = itr->first;
				uint32_t pos = itr->second.first;
				uint32_t docCount = itr->second.second;
				postingsFileList[batchIndex].seekg(sizeof(uint32_t) * pos * 2, std::ifstream::beg); // * 2 because every doc has docId and term frequency
				
				std::vector<std::pair<uint32_t, uint32_t> > postings;
				for (uint32_t j = 0; j < docCount; ++j) {
					uint32_t docId = 0;
					uint32_t tf = 0;
					postingsFileList[batchIndex].read((char*)&docId, 4);
					postingsFileList[batchIndex].read((char*)&tf, 4);
		
					postings.push_back(std::pair<uint32_t, uint32_t>(docId, tf));
				}
				batchIndexToPostings[batchIndex] = postings;

				batchIndexToWordIndex[batchIndex] += 1; // Advance the wordList after processing
			}

			// Merge and write postings to postings.bin, one docId at a time, in incremental order
			uint32_t currentDocId = 0;
			std::unordered_map<uint32_t, uint32_t> batchIndexToCurrentDocIdIndex;
			for (uint32_t i = 0; i < mergeFileCount; ++i) {
				batchIndexToCurrentDocIdIndex[i] = 0;
			}

			while (true) {
				// Check if all docIds of current word are processed
				bool finishCurrentWord = true;
				for (const auto& [j, postings] : batchIndexToPostings) {
					if (batchIndexToCurrentDocIdIndex[j] < postings.size()) {
						finishCurrentWord = false;
						break;
					}
				}
				if (finishCurrentWord)
					break;

				std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t> > > currentDocIdAndTfList; // [<batch_index, <docId, tf> >, ...]
				for (const auto& [j, postings] : batchIndexToPostings) {
					if (batchIndexToCurrentDocIdIndex[j] >= postings.size()) { // if a docIdList is finished, don't need to process it
						continue;
					}
					std::pair<uint32_t, uint32_t> docIdAndTf = postings[batchIndexToCurrentDocIdIndex[j]];
					if (currentDocId == 0 || docIdAndTf.first <= currentDocId) {
						if (docIdAndTf.first < currentDocId) {
							currentDocIdAndTfList.clear();
						}
						currentDocIdAndTfList.push_back(std::pair<uint32_t, std::pair<uint32_t, uint32_t> >(j, docIdAndTf));
						currentDocId = docIdAndTf.first;
					}
				}

				uint32_t tf = 0;
				for (auto itr = currentDocIdAndTfList.begin(); itr != currentDocIdAndTfList.end(); ++itr) {
					tf += itr->second.second; // Sum the term frequencies
					batchIndexToCurrentDocIdIndex[itr->first] += 1; // Advance the docIdList after processing
				}

				mergedPostingsFile.write((const char*)&currentDocId, 4);
				mergedPostingsFile.write((const char*)&tf, 4);

				// std::cout << "save <docId, tf>:" << currentDocId << " " << tf << std::endl;

				++docCounter;
				++currentWordDocCounter;

				currentDocId = 0; // Reset to 0, process next docId

			}

			// std::cout << "word:" << currentWord << "  currentWordDocCounter:" << currentWordDocCounter << std::endl;
			std::cout << currentWord << std::endl;

			allWordsData.push_back(std::pair<std::string, std::pair<uint32_t, uint32_t> >(currentWord, std::pair<uint32_t, uint32_t>(pos, currentWordDocCounter)));
			
			currentWordDocCounter = 0;
			currentWord = ""; // Reset to "", process next word
		}

		uint32_t wordCount = allWordsData.size();
		mergedWordsFile.write((const char*)&wordCount, 4); // 4 byte word count
		
		for (const auto& wordData : allWordsData) {
			std::string word = wordData.first;
			uint8_t wordLength = (uint8_t)word.length();
			mergedWordsFile.write((const char*)&wordLength, 1);
			mergedWordsFile.write(word.c_str(), wordLength);
			mergedWordsFile.write((const char*)&wordData.second.first, 4);
			mergedWordsFile.write((const char*)&wordData.second.second, 4);
		}
	}

	void addWordToPostings(const std::string& word, uint32_t docId) {
		// Since all documents are processed one by one, the current document is always the last one in postings.
		// So don't need wordToPostings.find(word), just access the last one
		std::vector<std::pair<uint32_t, uint32_t> >& postings = wordToPostings[word];
		if (postings.size() == 0 || postings[postings.size() - 1].first != docId) {
			postings.push_back(std::pair<uint32_t, uint32_t>(docId, 1));
		}
		else {
			postings[postings.size() - 1].second += 1;
		}
	}

	void runIndexer() {
		std::ifstream file(this->fileName);
		std::string line = "";

		bool readingTag = false;
		bool readingContent = false;

		size_t readStartIndex = 0;
		std::string currentTagName = "";
		std::string currentText = "";
		std::string currentDocNo = "";
		uint32_t currentDocumentLength = 0;

		uint32_t documentIndex = 0; // ++ when encounter </DOC>

		// Document offset table
		// For each document length,  offset(8 byte) + docNoLength(1 byte) + documentLength(how many bytes, not words)(2 bytes)
		std::ofstream docOffsetTableFile("index_docOffsetTable.bin");
		uint64_t docOffset = 0;

		// Document store
		// For each document, docNo(str) + document(str)
		std::ofstream documentsFile("index_documents.bin");

		while (getline(file, line)) {

			readStartIndex = 0;

			for (size_t i = 0; i < line.length(); ++i) {

				// Start of a tag
				if (line[i] == '<') {
					if (readingContent) {
						std::string content = line.substr(readStartIndex, i - readStartIndex);
						currentText += content;

						// Extract and print all the words
						std::vector<std::string> words = extractWords(currentText);

						if (currentTagName == "DOCNO") { // the '<' of </DOCNO>, the close tag of a document no.
							// currentDocNo = words[0]; //stripString(currentText); // Don't need to strip, extracted words are in good format
							currentDocNo = stripString(currentText);
							
							// Write document offset table
							docOffsetTableFile.write(reinterpret_cast<const char*>(&docOffset), sizeof(docOffset));
							uint8_t docNoLength = currentDocNo.length();
							docOffsetTableFile.write(reinterpret_cast<const char*>(&docNoLength), sizeof(docNoLength));
							docOffset += docNoLength;

							// Write docNo to document store
							documentsFile.write(currentDocNo.c_str(), currentDocNo.size());
						}

						// Output the words, and save to postings
						for (size_t wordIndex = 0; wordIndex < words.size(); ++wordIndex) {
							std::string word = words[wordIndex];
							// std::cout << "token:" << word << std::endl; // output each word as a line

							uint32_t docId = documentIndex + 1;
							this->addWordToPostings(word, docId);
						}

						// Save document length
						currentDocumentLength += words.size();

						readingContent = false;
					}

					// Start to read tag
					readingTag = true;
					readStartIndex = i + 1;

					continue;
				}

				// End of a tag
				if (line[i] == '>') {
					if (readingTag) {
						std::string tagName = line.substr(readStartIndex, i - readStartIndex);
						if (tagName[0] != '/') { // It's an open tag. e.g. <DOC>
							currentTagName = tagName;
						}
						else { // It's a close tag. e.g. </DOC>

							// Reach the end of </TEXT>
							if (tagName == "/TEXT") {
								// Write document offset table
								uint16_t textLength = currentText.length();
								docOffsetTableFile.write(reinterpret_cast<const char*>(&textLength), sizeof(textLength));
								docOffset += textLength;

								// Write document content to document store
								documentsFile.write(currentText.c_str(), currentText.size());
							}

							// Reach the end of a document
							if (tagName == "/DOC") { 
								currentDocNo = "";

								// Save current document length
								this->documentLengthList.push_back(currentDocumentLength);
								currentDocumentLength = 0;
								
								// Output an blank line between documents
								// std::cout << std::endl;

								// Save batch postings
								++this->postingsBatchCounter;
								if (this->postingsBatchCounter >= this->postingsBatchSize) {
									this->postingsBatchCounter = 0;
									this->savePostingBatch();
								}

								if (documentIndex % 1000 == 0) 
								{
									std::cout << documentIndex << " documents processed." << std::endl;
								}
								++documentIndex;
							}

							currentTagName = "";
							currentText = "";
						}
						
						readingTag = false;
					}

					// Start to read content
					readingContent = true;
					readStartIndex = i + 1;

					continue;
				}
			}

			// Add the rest of the line to currentText
			if (readingContent) {
				std::string content = line.substr(readStartIndex, line.length() - readStartIndex);
				if (content.length() > 0)
					currentText += content + "\n";
			}
		}
		
		// Save the last batch
		this->savePostingBatch();

		std::cout << "Merging postings..." << std::endl;
		this->mergePostingsBatch();

		std::cout << "All " << documentIndex << " documents processed." << std::endl;

		this->saveIndexToFiles();

		std::cout << "Saved to index files." << std::endl;
	}
};

int main(int argc, char* argv[]) {
	// if (argc != 2) {
	// 	std::cout << "Usage: enter a parameter as the file to create index. Example: ./indexer wsj.xml" << std::endl;
	// 	return 0;
	// }

	// Indexer indexer(argv[1]);

	Indexer indexer("PubMed/PubMed_abstract_100.xml");
	indexer.runIndexer();

	return 0;
}
