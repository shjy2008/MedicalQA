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

	// DOCNO list 
	// e.g. [WSJ870324-0001, WSJ870323-0181, ...]
	std::vector<std::string> docNoList;

	// Document length list
	// e.g. [159, 64, 48, 30, 106, 129, ...]
	std::vector<uint32_t> documentLengthList;

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

		// Save DOCNO list
		std::ofstream docNoFile("index_docNo.bin"); // docNo("WSJ870324-0001") string splitted by \0
		for (size_t i = 0; i < this->docNoList.size(); ++i) {
			std::string s = this->docNoList[i] + '\0'; // Add '\0' to the end to split strings
			docNoFile.write(s.c_str(), s.length());
		}

		// Save wordToPostings
		// Stored as (docId1 for word1, term frequency 1 for word1, docId2 for word1, tf2 for word1, 
		// 				docId1 for word2, tf1 for word2, ...) each in 4 bytes uint32_t
		std::ofstream wordPostingsFile("index_wordPostings.bin");
		
		// Stored as: 4 byte word count + [(wordLength(1 byte), word, pos(4 bytes), docCount(4 bytes)), ...]
		// -- pos: how many documents before the word's first document
		// -- docCount: how many documents the word appears in (vector's size) 
		std::ofstream wordsFile("index_words.bin");

		uint32_t wordCount = (uint32_t)wordToPostings.size();
		wordsFile.write((const char*)&wordCount, 4); // 4 byte word count

		uint32_t docCounter = 0;
		for (std::unordered_map<std::string, std::vector<std::pair<uint32_t, uint32_t> > >::iterator it 
					= wordToPostings.begin(); it != wordToPostings.end(); ++it) 
		{
			std::string word = it->first;
			std::vector<std::pair<uint32_t, uint32_t> > postings = it->second;
			uint32_t docCount = postings.size();

			uint8_t wordLength = (uint8_t)word.length();
			wordsFile.write((const char*)&wordLength, 1);
			wordsFile.write(word.c_str(), wordLength);
			wordsFile.write((const char*)&docCounter, 4);
			wordsFile.write((const char*)&docCount, 4);

			for (uint32_t i = 0; i < docCount; ++i) {
				uint32_t docId = postings[i].first;
				uint32_t tf = postings[i].second;
				wordPostingsFile.write((const char*)&docId, 4);
				wordPostingsFile.write((const char*)&tf, 4);

				++docCounter;
			}
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
							currentDocNo = words[0]; //stripString(currentText); // Don't need to strip, extracted words are in good format
							this->docNoList.push_back(currentDocNo);
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

							// Reach the end of a document
							if (tagName == "/DOC") { 
								currentDocNo = "";

								// Save current document length
								this->documentLengthList.push_back(currentDocumentLength);
								currentDocumentLength = 0;


								// Output an blank line between documents
								// std::cout << std::endl;

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
				currentText += content + "\n";
			}
		}

		std::cout << "All " << documentIndex << " documents processed." << std::endl;

		this->saveIndexToFiles();

		std::cout << "Saved to index files." << std::endl;
	}
};

int main(int argc, char* argv[]) {
	if (argc != 2) {
		std::cout << "Usage: enter a parameter as the file to create index. Example: ./indexer wsj.xml" << std::endl;
		return 0;
	}

	Indexer indexer(argv[1]);
	indexer.runIndexer();

	return 0;
}
