import shutil
import os
test_list = [
    # 'trainset_split_largebox/冀-廊坊-百世-分拨-市区主线_20210829205340-20210829205410_1_largebox/冀-廊坊-百世-分拨-市区主线_20210829205340-20210829205410_1_largebox',
    # 'trainset_split_largebox/琼-海口-圆通-装卸区2_0118009034_20220120224840_0_largebox/琼-海口-圆通-装卸区2_0118009034_20220120224840_0_largebox',
    # 'trainset_split_largebox/351W-晋-太原-顺丰转运-全景-F区08_0900000011_20220111171339_0_largebox/351W-晋-太原-顺丰转运-全景-F区08_0900000011_20220111171339_0_largebox',
    # 'trainset_split_largebox/贵_贵阳_京东_三线左面发货2_0115181003_20220111162621_0_largebox/贵_贵阳_京东_三线左面发货2_0115181003_20220111162621_0_largebox',
    # 'trainset_split_largebox/冀-秦皇岛-中通-秦皇岛33510-出港大件卸货口_0118043631_20220124160105_0_largebox/冀-秦皇岛-中通-秦皇岛33510-出港大件卸货口_0118043631_20220124160105_0_largebox',
    # 'trainset_split_largebox/暴力分拣 贵_贵阳_京东_遵义发货_0115181022_20220111152203_0_largebox/暴力分拣 贵_贵阳_京东_遵义发货_0115181022_20220111152203_0_largebox',
    # 'trainset_split_largebox/暴力分拣 贵_贵阳_京东_遵义发货1_0115181022_20220111152056_0_largebox/暴力分拣 贵_贵阳_京东_遵义发货1_0115181022_20220111152056_0_largebox',
    # 'trainset_split_largebox/山东省-泰安市-邮政-装卸区-12暴力_0901011012_20220111170824_0_largebox/山东省-泰安市-邮政-装卸区-12暴力_0901011012_20220111170824_0_largebox',
    # 'trainset_split_largebox/（双人扔抛大件）琼-海口-极兔-建包区3-IDF2-092_0118005871_20220208215957_0_largebox/（双人扔抛大件）琼-海口-极兔-建包区3-IDF2-092_0118005871_20220208215957_0_largebox',
    # 'trainset_split_largebox/晋-太原-申通-分拣区-4_0100053011_20220126113634_0_largebox/晋-太原-申通-分拣区-4_0100053011_20220126113634_0_largebox',
    # 'trainset_split_largebox/晋-太原-韵达转运-分拣5_0100060016_20220126130658_0_largebox/晋-太原-韵达转运-分拣5_0100060016_20220126130658_0_largebox',
    # 'trainset_split_largebox/亳州 Camera 01_0602870997_20220126143001_0_largebox/亳州 Camera 01_0602870997_20220126143001_0_largebox',
    # 'trainset_split_largebox/（暴力分拣）川-成都-申通进分拣 青龙场_0100050031_20220222231623_0_largebox/（暴力分拣）川-成都-申通进分拣 青龙场_0100050031_20220222231623_0_largebox',
    # 'trainset_split_largebox/（暴力分拣）川-成都-申通-土桥_安德_0100050030_20220222231350_0_largebox/（暴力分拣）川-成都-申通-土桥_安德_0100050030_20220222231350_0_largebox',
    # 'trainset_split_largebox/（暴力分拣）川-成都-申通-土桥_安德_0100050030_20220222231447_0_largebox/（暴力分拣）川-成都-申通-土桥_安德_0100050030_20220222231447_0_largebox',
    # 'trainset_split_largebox/（暴力分拣）川-成都-申通-土桥_安德_0100050030_20220222231945_0_largebox/（暴力分拣）川-成都-申通-土桥_安德_0100050030_20220222231945_0_largebox',
    # 'trainset_split_largebox/（暴力分拣）川-成都-申通-土桥_安德_0100050030_20220222232350_0_largebox/（暴力分拣）川-成都-申通-土桥_安德_0100050030_20220222232350_0_largebox',
    # 'trainset_split_largebox/（暴力分拣）晋-大同-中通转运-分拣区-2_62564_largebox/（暴力分拣）晋-大同-中通转运-分拣区-2_62564_largebox',
    # 'trainset_split_largebox/（暴力分拣）辽-大连-韵达-海事二七_1646201880943_largebox/（暴力分拣）辽-大连-韵达-海事二七_1646201880943_largebox',
    # 'trainset_split_largebox/（暴力分拣）琼-海口-极兔-DWS5号后-IDF2-082_0118005881_20220218201018_0_largebox/（暴力分拣）琼-海口-极兔-DWS5号后-IDF2-082_0118005881_20220218201018_0_largebox',
    # 'testset_split_largebox/鄂-武汉-申通-武汉分拣区-6_0100050020_20220112180109_0_largebox/鄂-武汉-申通-武汉分拣区-6_0100050020_20220112180109_0_largebox',
    # 'testset_split_largebox/鄂-武汉-申通-武汉转运安检区1-1_0100050003_20220121154821_0_largebox/鄂-武汉-申通-武汉转运安检区1-1_0100050003_20220121154821_0_largebox',
    # 'testset_split_largebox/冀-廊坊-百世-分拨-北京短驳_20210829200238-20210829200259_1_largebox/冀-廊坊-百世-分拨-北京短驳_20210829200238-20210829200259_1_largebox',
    # 'testset_split_largebox/冀-石家庄-韵达-石家庄-C3-021备装安检_0100060013_20220111205910_0_largebox/冀-石家庄-韵达-石家庄-C3-021备装安检_0100060013_20220111205910_0_largebox',
    # 'testset_split_largebox/晋-太原-申通-安检区-1_0100053002_20220120223110_0_largebox/晋-太原-申通-安检区-1_0100053002_20220120223110_0_largebox',
    'testset_split_largebox/（暴力分拣）川-成都-申通-土桥_安德0100050030_20220222232259_0_0_largebox/（暴力分拣）川-成都-申通-土桥_安德0100050030_20220222232259_0_0_largebox',
    # 'testset_split_largebox/（暴力分拣）川-成都-申通-土桥_安德_0100050030_20220222232427_0_largebox/（暴力分拣）川-成都-申通-土桥_安德_0100050030_20220222232427_0_largebox',
    # 'testset_split_largebox/（踩踏-抛扔）琼-海口-极兔-建包区3-IDF2-092_0118005871_20220214210416_0_largebox/（踩踏-抛扔）琼-海口-极兔-建包区3-IDF2-092_0118005871_20220214210416_0_largebox',
    # 'testset_split_largebox/（踩踏-抛扔）琼-海口-极兔-建包区3-IDF2-092_0118005871_20220214210709_0_largebox/（踩踏-抛扔）琼-海口-极兔-建包区3-IDF2-092_0118005871_20220214210709_0_largebox',
]

dst_file = '/home/haoren/repo/slowfast/data/raw/anno_seg_test/test.csv'

for test_cur in test_list:
    print(test_cur)
    src_file = '/home/haoren/repo/slowfast/data/raw/test_acc/' + test_cur + '.txt'
    shutil.copy(src_file, dst_file)
    os.system('bash test_seg.sh')
