先端人工知能という授業での研究成果をおいておくレポジトリ。ポスターは[こちら](https://github.com/minnsou/FrontierArtificialIntelligence/blob/master/poster.pdf)。

- mahjong.ipynb kerasを用いた麻雀のあがり判定の学習。手牌を入力として、あがりか否かの二値分類を教師あり学習をする。

- mahjong_pytorch.py pytorchを用いた麻雀のあがり判定の学習。基本的には上と同じ。

- mahjong_utils.py 麻雀の手牌生成やあがり判定などの関数を置いておくpythonファイル。

- poker.ipynb ポーカーにおける役判定の学習。手牌を入力として、役が成立するか否かの二値分類の教師あり学習をする。

- data 学習に用いる手牌のデータを保存するディレクトリ。麻雀で使うデータもポーカーで使うデータも両方入れている。ただし、ポーカーで必要なhand_list_200000.npyは容量が大きすぎてGithubには置けなかった。

- models 学習結果を保存するディレクトリ。今のところ麻雀の結果のみ。

- draw_pdf 学習結果の図を保存するディレクトリ。

- poster.pdf ポスター発表で使ったポスター。
