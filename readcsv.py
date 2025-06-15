import csv

# Specify the path to your CSV file
csv_file_path = '/dataT1/Free/qinan/intestine/extractValid_fold0_traindata .csv'
text_file_path = '/dataT1/Free/qinan/intestine/train_path_list-new.txt'

# Initialize an empty list to store the data
data = []

# Open the CSV file
with open(text_file_path, 'w') as txtfile:
    with open(csv_file_path, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)

        # Skip the header row if it exists
        header = next(csv_reader)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Append the row to the data list
        
            if int(row[2]) < 100:

                if 'TYH' in row[1] :
                    #print('row[1] = ',row[1])
                    txtfile.write('/data4/CTInfa/Ileus/IleusTYH/dataset20230509_withAMU/interp/'+row[1].replace('.nii','.nii.gz')+' '\
                                '/dataT1/Free/qinan/intestine/label/'+row[1].replace('.nii','-labels.nii.gz')+'\n')
                if 'AMU' in row[1]:
                 
                    txtfile.write('/data4/CTInfa/Ileus/IleusTYH/dataset20230509_withAMU/interp/'+row[1].replace('.nii','.nii.gz')+' '\
                                '/dataT1/Free/qinan/intestine/label/'+row[1][:-10]+'.nii.gz'+'\n')
            else:
                continue


            #print('row = ', row)
            #print('row[2] = ',row[2])
            #print('row[1] = ',row[1])
            #1/0
            #data.append(row)
            #print('data = ', data)
            #1/0

# Print the data
#for row in data:
#    print(row)
