file_to_count = open('elton_john_songs_lyrics.csv').read()
split_file = file_to_count.split()
unique_words = set(w.lower() for w in split_file)
unique_words_len = len(unique_words)
print unique_words_len