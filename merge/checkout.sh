# dev
rm -rf branch-dev && git clone git@gitlab.com:chuong98vt/ccdetpose --branch dev --single-branch --depth 1 ./branch-dev
rm -rf branch-master && git clone git@gitlab.com:chuong98vt/ccdetpose --branch master --single-branch --depth 1 ./branch-master
# cc
# git checkout master ccdetection
# # mm
cd branch-master
rm -rf mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git reset --hard 4d84161f142b7500089b0db001962bbc07aa869d
