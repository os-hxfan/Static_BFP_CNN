input="./groundtruth.txt"
while read -r var
do
  line="$var"
  line=($line)
  if [ -f $($PWD/${line[1]}) ]
  then
    #echo "mv"
    echo $(mv ${line[0]} ${line[1]})
  else
    echo $(mkdir ${line[1]})
    echo $(mv ${line[0]} ${line[1]}) 
    echo "create new folder ${line[1]}"
  fi
done < "$input"
