import psycopg2
import pickle
from open_images import OpenImages

def pickle_stuff(filename, data):
    ''' open file '''
    with open(filename, 'w') as picklefile:
        pickle.dump(data, picklefile)

def unpickle(filename):
    ''' save file '''
    with open(filename, 'r') as picklefile:
        old_data = pickle.load(picklefile)
    return old_data

def make_classes_table(cur, conn):
    cur.execute("CREATE TABLE stl_classes ( class_id serial PRIMARY KEY, class_name VARCHAR (50) UNIQUE NOT NULL);")
    conn.commit()

def populate_classes_table(cur, conn, oi):
    class_names = oi.get_class_names()
    for cn in class_names:
        cur.execute("INSERT INTO stl_classes VALUES ( %d, '%s' );" %cn)
    conn.commit()

def make_images_table(cur, conn):
    cur.execute("CREATE TABLE stl_images ( byte_index integer, class_id integer, size integer, train_test VARCHAR (50));")
    conn.commit()

def populate_images_table(cur, conn, oi, train_test):
    size = oi.IMCHSIZE
    data = oi.get_all_labels()
    for d in data:
        vals = (d[0], d[1], size, train_test)
        cur.execute("INSERT INTO stl_images VALUES ( %d, %d, %d, '%s');" %vals)
    conn.commit()

def make_folds_table(cur, conn):
    cur.execute("CREATE TABLE stl_folds ( byte_index integer, fold_num integer );")
    conn.commit()

def populate_fold_table(cur, conn, oi):
    fold_data = oi.get_all_image_folds()
    for fd in fold_data:
        cur.execute("INSERT INTO stl_folds VALUES ( %d, %d );" %fd)
    conn.commit()
    
            
if __name__ == "__main__":
    trainsoi = OpenImages()
    testsoi = OpenImages(xfl="test_X.bin", yfl="test_y.bin")

    params = unpickle("params.pkl")
    
    conn = psycopg2.connect(**params)
    cur = conn.cursor()
    
    #make_classes_table(cur, conn)
    #populate_classes_table(cur, conn, trainsoi)
    #make_images_table(cur, conn)
    #make_folds_table(cur, conn)
    #populate_images_table(cur, conn, trainsoi, "train")
    #populate_images_table(cur, conn, testsoi, "test")
    #populate_fold_table(cur, conn, trainsoi)

    cur.close()
    conn.close()

    '''
    select byte_index, count(*) from stl_folds where byte_index > 4900 
        group by byte_index order by count desc limit 10;
        
    select fold_count, count(*) as image_count from (
           select byte_index, count(*) as fold_count from stl_folds 
           group by byte_index)
        as count_table group by fold_count order by image_count desc;
    
     fold_count | image_count 
    ------------+-------------
              1 |        2071
              2 |        1482
              3 |         968
              4 |         359
              5 |         101
              6 |          13
              7 |           6
    '''
