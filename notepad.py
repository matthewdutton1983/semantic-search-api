count = 0
    
    unique_sentences = {}
    seen_sentences = {}
    
    batch_sentences = []
    batch_sent_idx = []
    batch_embeddings = []
    
    BATCH_SIZE = 10000

    token = get_doclink_token(username, password)

    for unique_id in unique_ids:
        try:
            if count % BATCH_SIZE == 0:
                token = get_doclink_token(username, password)
                logging.info(f"New token fetched for batch starting at {count}.")

                document_text = get_document_text(unique_id, token)
                document_name = get_document_name(unique_id, token)

                raw_sentences = sent_tokenize(document_text)

                for raw_sentence in raw_sentences:
                    clean_sentence = raw_sentence.lower()

                    if not clean_sentence.strip():
                        continue

                    if clean_sentence not in seen_sentences:
                        embedding = embed([clean_sentence]).numpy()[0].tolist()
                        batch_embeddings.append(embedding)

                        unique_sentences[count] = {
                            "original_text": raw_sentence,
                            "clean_text": clean_sentence,
                            "document_ids": [unique_id],
                            "embedding": embedding
                        }

                        batch_sent_idx.append(count)
                        seen_sentences[clean_sentence] = count

                        batch_sentences.append({
                            "sent_idx": count,
                            "original_text": raw_sentence,
                            "clean_text": clean_sentence,
                            "document_ids": [unique_id],
                            "embedding": embedding
                        })
                    else:
                        sentence_index = seen_sentences[clean_sentence]
                        unique_sentences[sentence_index]["document_ids"].append(unique_id)
                    
                    count += 1

                    if count % BATCH_SIZE == 0:
                        with session.begin():
                            session.execute(sentences_table.insert(), batch_sentences)
                            faiss_index.add_with_ids(np.array(batch_embeddings), np.array(batch_sent_idx))
                            batch_sentences = []
                            batch_sent_idx = []
                            batch_embeddings = []
                            logging.info(f"Loaded {count} sentences.")
        except Exception as e:
            logging.error(f"Error processing document with id {unique_id}: {str(e)}")

    if batch_sentences:
        with session.begin():
            session.execute(sentences_table.insert(), batch_sentences)
            faiss_index.add_with_ids(np.array(batch_embeddings), np.array(batch_sent_idx))

    faiss.write_index(faiss_index, "sentences.faiss")
