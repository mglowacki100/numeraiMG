# m - parameters for classifier
# d - extra dimensions (features) in train/test files
for m  in 1 #2 3 #3 4 5 6 7 8 9 10 11 2
do
  for d in 16 17 #13 14 15 #9 10 11 12  #0 4 5 6 7 8
  do
    python nai_paribas_m_d.py $m $d >> "output/_${m}_$d.txt"
  done 
done

