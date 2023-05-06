# Dataset Description
1. `chicago/act_ext` contains the 2019 taxi trip data of region `ID=i` in Chicago.
   1. E.g., `taxi1.txt=taxi-trip data of Rogers Park(ID=1)`.
2. `chicago/com_crime` contains the 2019 crime data of community `ID=i-1` in Chicago.
   1. E,g., `r_0.txt=crime-data of Rogers Park (ID=1)`.
   2. Each row in `r_0.txt` shows the crime occurrences of crime category `ID=k` (See `chicago/chicago_crime-cat_to_id.txt`)
3. `chicago/side_crime` contains the 2019 crime data of the sides in Chicago.
4. `chicago/chicago_cid_to_name.txt` maps the community-IDs to community-names.
5. `chicago/chicago_crime-cat_to_id.txt` maps the crime categories to crime category IDs.
6. `chicago/poi.txt` contains the POI information of Chicago communities. 
7. `chicago/side_com_adj.txt` maps the Chicago communities to its corresponding side.
   1. `<SIDE-ID: ID1> <COM-ID:ID2>:` Side `SIDE-ID` contains community `COM-ID`
   2. The mapping between the side-ids of `chicago/side_com_adj.txt(=SID1)` and `chicago/side_crime(=SID2)` is follows: `SID2=SID1%101`