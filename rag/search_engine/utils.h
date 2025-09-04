
#include <stdint.h>
#include <string>

struct Posting {
	uint32_t docId;
	uint32_t tf;

    Posting() : docId(0), tf(0) {}

    Posting(uint32_t docId, uint32_t tf) : docId(docId), tf(tf) {}
};

struct WordData {
    uint32_t postingsPos; // In the postings file, how many documents before the word's postings
    uint32_t postingsDocCount; // In the postings file, how many documents the word's postings have
    float impactScore; // max impact score the word can have

    WordData() : postingsPos(0), postingsDocCount(0), impactScore(0) {}
    WordData(uint32_t postingsPos, uint32_t postingsDocCount) : postingsPos(postingsPos), postingsDocCount(postingsDocCount), impactScore(0) {}
    WordData(uint32_t postingsPos, uint32_t postingsDocCount, float impactScore) : postingsPos(postingsPos), postingsDocCount(postingsDocCount), impactScore(impactScore) {}
};

struct DocumentOffset {
    uint64_t offset;
    uint8_t docNoLength;
    uint16_t documentLength;

    DocumentOffset(uint64_t offset, uint8_t docNoLength, uint16_t documentLength) : offset(offset), docNoLength(docNoLength), documentLength(documentLength) {}
};

struct SearchResult {
	uint32_t docId;
	float score;

    std::string docNo;
    std::string content;

	SearchResult(uint32_t docId, float score) : docId(docId), score(score) {}

	bool operator > (const SearchResult& other) const {
		return score > other.score; // min-heap based on score
	}
};

const std::vector<std::string> stopWords = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now",
    "ll", "ve", "m", "d", "re"
};

class Utils {
	public:

    constexpr const static float bm25_k1 = 1.2f;
    constexpr const static float bm25_b = 0.75f;

    // Okapi BM25 https://en.wikipedia.org/wiki/Okapi_BM25
    static inline float getIDF(uint32_t docCountContainWord, uint32_t totalDocuments) {
        return std::log((totalDocuments - docCountContainWord + 0.5) / (docCountContainWord + 0.5) + 1); // Ensure positive
    }

	// tf_td: number of the term appears in doc
	// docLength: how many words in the document
	// idf: inverted document frequency (calculated by total document and documents contain the word)
    static inline float getRankingScore(uint32_t tf_td, uint32_t docLength, float idf, float averageDocumentLength) {
		// TF-IDF
		// float tf_td_normalized = (float)tf_td / docLength;
		// float idf = (float)this->totalDocuments / docCountContainWord;
		// return tf_td_normalized * idf;

		// BM25 - in the slides (but it will produce negative value when docCountContainWord > 1/2 totalDocuments, then the ranking is wrong)
		// float w_t = std::log2f((this->totalDocuments - docCountContainWord + 0.5f) / (docCountContainWord + 0.5f));
		// float k1 = 1.2f;
		// float k3 = 7;
		// float b = 0.75f;
		// float K = k1 * ((1 - b) + (b * docLength / this->averageDocumentLength));
		// float w_dt = w_t * ((k1 + 1) * tf_td / (K + tf_td)) * ((k3 + 1) * tf_tq / (k3 + tf_tq));
		// return w_dt;

		// Okapi BM25 https://en.wikipedia.org/wiki/Okapi_BM25
		float K = bm25_k1 * ((1 - bm25_b) + bm25_b * (docLength / averageDocumentLength));
		float score = idf * (tf_td * (bm25_k1 + 1) / (tf_td + K));
		return score;
	}

    // Extract words from a text string
    static std::vector<std::string> extractWords(const std::string& text) {
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
    static std::string stripString(const std::string& text) {
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
    
};