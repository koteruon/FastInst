file_path = r'/home/chenzy/FastInst-main/output/video_output/IMG7408_R_paddle.txt'
f = open(file_path, 'r')
data = f.read()
list_data = data.split(',')
print(list_data)

out_path = r'/home/chenzy/FastInst-main/output/data_test_1_10f.txt'
out_label_2cls_path = r'/home/chenzy/FastInst-main/output/data_test_1_label_10f.txt'
out_label_3cls_path = r'/home/chenzy/FastInst-main/output/data_test_1_label_10f_3cls.txt'
out_f = open(out_path,"a")
out_label_f_2cls = open(out_label_2cls_path,"a")
out_label_f_3cls = open(out_label_3cls_path,"a")

print(len(list_data))
count = 0
for i in range(len(list_data)):
    if count!= 10:
        out_f.write(list_data[i]+" ")
        count+=1
    else:
        out_f.write("\n")
        out_f.write(list_data[i]+" ")
        count = 0
        count+= 1 
        out_label_f_2cls.write("1")
        out_label_f_2cls.write("\n")
        out_label_f_3cls.write("2")
        out_label_f_3cls.write("\n")
        # out_label_f.write("1")
        # out_label_f.write("2")


