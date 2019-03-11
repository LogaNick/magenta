#!/bin/bash

DIR=$1

# Array of control signals to iterate over
declare -a signatures
signatures[0]="1873RachmaninoffRussia"
signatures[1]="1810ChopinPoland"
signatures[2]="1862Debussy"
signatures[3]="1732Haydn"
signatures[4]="1685Bach"
signatures[5]="1860Albeniz"
signatures[6]="1811Liszt"
signatures[7]="1700Russia"
signatures[8]="1800USA"
signatures[9]="1950France"
signatures[10]="1800Tunisia"
signatures[11]="1800Estonia"
signatures[12]="1800Kazakhstan"
signatures[13]="1850NorthAtlantic"
signatures[14]="1850Cyprus"
signatures[15]="1800NorthPole"
signatures[16]="1850SouthAfrica"

i=0
j=0


for file in `ls $DIR*`
do
	NAME=$( dirname $file )
	echo $file"\t\t${i}"
	#echo $NAME'/'"${signatures[$j]}"'_'$(( i%2 + 1 ))'.mid'
	#mv $file $NAME'/'"${signatures[$j]}"'_'$(( i%2 + 1 ))'.mid'

	(( i++ ))

	if (( i%2 == 0 )); then
		(( j++ ))
	fi
done

