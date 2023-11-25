# Water-Quality

Hello World

hello khomsan

หา index ใน maaxscores
หา model ที่ดีที่สุด
m=max(scoresl['test_accuracy'])
a=scoresl['test_accuracy']
max_indices = np.where(a == m)[0]
print('test_accuracy max',max_indices)

เลือก model ที่ดีที่สุด
allmodels = scoresl['estimator']
models = allmodels[int(max_indices)]
print(models)

else :
            n=user_name
            if n == '':
                st.write("Enter name model")
            else:
                partname=f"./models/"+n+".pkl"
                st.write(st.session_state.save_model)
                conn = sqlite3.connect('.\pages\DB\pathmodels.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO pathmodels (name,path,testaccuracy,trainaccuracy) VALUES (?,?,?,?)
                ''',(n,partname,test_accuracy,train_accuracy))
                conn.commit()
                conn.close()
                dump(model,partname )
