import csv

artist_class_21_images = set()
with open('data/artist_val.csv', 'r', newline='') as art_file:
    reader = csv.reader(art_file)
    for row in reader:
        image_path, artist_class = row
        if int(artist_class) == 21:
            artist_class_21_images.add(image_path)

genre_with_dali = []
genre_without_dali = []

with open('data/genre_val.csv', 'r', newline='') as genre_file:
    reader = csv.reader(genre_file)
    for row in reader:
        image_path, genre_class = row
        if image_path in artist_class_21_images:
            genre_with_dali.append(row)
        else:
            genre_without_dali.append(row)

with open('data/genre_val_with_dali.csv', 'w', newline='') as file1:
    writer = csv.writer(file1)
    writer.writerows(genre_with_dali)

with open('data/genre_val_without_dali.csv', 'w', newline='') as file2:
    writer = csv.writer(file2)
    writer.writerows(genre_without_dali)