import os
import csv

def create_folder(folder_path):
  if not os.path.exists(folder_path):
    try:
      os.makedirs(folder_path)
      print(f"Successfully created folder: {folder_path}")
    except OSError as error:
      print(f"Error creating folder: {folder_path}. Reason: {error}")


def generate_tsv(data_path, country, output_path, tweet_file_name="test_text.txt", label_file_name="test_labels.txt", output_file_name="tweets_test_{}.tsv"):

  country_path = os.path.join(data_path, country)

  if not os.path.exists(country_path):
    print(f"Skipping {country}: Directory not found")
    return

  # Define file paths within the country directory
  tweet_file = os.path.join(country_path, tweet_file_name)
  label_file = os.path.join(country_path, label_file_name)
  output_file_name_country = output_file_name.format(country)
  output_file = os.path.join(output_path, output_file_name_country)


  create_folder(output_path)
  # Generate the TSV file if data files exist
  if os.path.exists(tweet_file) and os.path.exists(label_file):
    with open(tweet_file, 'r') as tweet_data, open(label_file, 'r') as label_data, open(output_file, 'w', newline='') as tsvfile:
      # Create CSV writer with tab delimiter
      writer = csv.writer(tsvfile, delimiter='\t')

      # Write header row
      writer.writerow(['tweet', 'label'])

      # Read data line by line and write to TSV
      for tweet_line, label_line in zip(tweet_data, label_data):
        # Remove trailing newline characters
        tweet = tweet_line.strip()
        label = label_line.strip()
        writer.writerow([tweet, label])

    print(f"Successfully created TSV for {country}: {output_file}")
  else:
    print(f"Skipping {country}: Missing data files")

output_path = "tsv_data_xlmt"
data_path = "xlm-t/data/sentiment/"
countries = ["arabic",
             "english",
             "french",
             "german",
             "hindi",
             "italian",
             "portuguese",
             "spanish"]

for country in countries:
  generate_tsv(data_path, country, output_path)
