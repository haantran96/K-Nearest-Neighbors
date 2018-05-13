

```python
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, metrics, svm

songs_dataset = pd.read_json('MasterSongList.json')
```


```python
songs_dataset.loc[:,'genres'] = songs_dataset['genres'].apply(''.join)
def consolidateGenre(genre):
    if len(genre)>0:
        return genre.split(':')[0]
    else: return genre

songs_dataset.loc[:, 'genres'] = songs_dataset['genres'].apply(consolidateGenre)
```


```python
audio_feature_list = [audio_feature for audio_feature in songs_dataset['audio_features']]
audio_features_headers = ['key','energy','liveliness','tempo','speechiness','acousticness','instrumentalness','time_signature'
                         ,'duration','loudness','valence','danceability','mode','time_signature_confidence','tempo_confidence'
                         ,'key_confidence','mode_confidence']
audio_features = pd.DataFrame(audio_feature_list, columns=audio_features_headers)
audio_features.loc[:,].dropna(axis=0,how='all',inplace=True)
audio_features['genres'] = songs_dataset['genres']
```

## Unbalanced data


```python
rock_rap = audio_features.loc[(audio_features['genres'] == 'rock') | (audio_features['genres'] == 'rap')]
rock_rap.reset_index(drop=True)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>energy</th>
      <th>liveliness</th>
      <th>tempo</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>time_signature</th>
      <th>duration</th>
      <th>loudness</th>
      <th>valence</th>
      <th>danceability</th>
      <th>mode</th>
      <th>time_signature_confidence</th>
      <th>tempo_confidence</th>
      <th>key_confidence</th>
      <th>mode_confidence</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11.0</td>
      <td>0.398953</td>
      <td>0.170640</td>
      <td>77.821</td>
      <td>0.033343</td>
      <td>0.007855</td>
      <td>0.457147</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>548.03156</td>
      <td>-19.753</td>
      <td>0.243535</td>
      <td>0.198917</td>
      <td>0.901</td>
      <td>0.676</td>
      <td>0.362</td>
      <td>0.996</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>0.906388</td>
      <td>0.130576</td>
      <td>127.438</td>
      <td>0.122818</td>
      <td>0.000353</td>
      <td>0.001200</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>210.42667</td>
      <td>-5.856</td>
      <td>0.318454</td>
      <td>0.418888</td>
      <td>0.407</td>
      <td>0.602</td>
      <td>0.603</td>
      <td>0.889</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.0</td>
      <td>0.682443</td>
      <td>0.163735</td>
      <td>146.216</td>
      <td>0.183496</td>
      <td>0.030170</td>
      <td>0.000033</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>222.58667</td>
      <td>-6.162</td>
      <td>0.531741</td>
      <td>0.766924</td>
      <td>0.587</td>
      <td>0.342</td>
      <td>0.064</td>
      <td>0.982</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.0</td>
      <td>0.699017</td>
      <td>0.041737</td>
      <td>140.624</td>
      <td>0.038933</td>
      <td>0.158435</td>
      <td>0.000080</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>471.20000</td>
      <td>-7.392</td>
      <td>0.725072</td>
      <td>0.439839</td>
      <td>0.594</td>
      <td>0.542</td>
      <td>0.133</td>
      <td>0.691</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.0</td>
      <td>0.722723</td>
      <td>0.074246</td>
      <td>89.999</td>
      <td>0.340797</td>
      <td>0.260922</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>193.56000</td>
      <td>-2.801</td>
      <td>0.807143</td>
      <td>0.907577</td>
      <td>0.217</td>
      <td>0.231</td>
      <td>0.603</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>0.870259</td>
      <td>0.484105</td>
      <td>110.075</td>
      <td>0.042160</td>
      <td>0.003531</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>187.52000</td>
      <td>-4.185</td>
      <td>0.221896</td>
      <td>0.488180</td>
      <td>0.181</td>
      <td>0.475</td>
      <td>0.782</td>
      <td>0.793</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2.0</td>
      <td>0.738028</td>
      <td>0.091445</td>
      <td>119.004</td>
      <td>0.030686</td>
      <td>0.157794</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>235.28000</td>
      <td>-3.918</td>
      <td>0.317212</td>
      <td>0.610500</td>
      <td>0.376</td>
      <td>0.579</td>
      <td>0.784</td>
      <td>0.950</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.0</td>
      <td>0.830646</td>
      <td>0.290238</td>
      <td>102.967</td>
      <td>0.049160</td>
      <td>0.000291</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>211.38667</td>
      <td>-3.957</td>
      <td>0.714192</td>
      <td>0.676519</td>
      <td>0.377</td>
      <td>0.342</td>
      <td>0.930</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.0</td>
      <td>0.832268</td>
      <td>0.273200</td>
      <td>164.070</td>
      <td>0.059476</td>
      <td>0.061904</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>217.66667</td>
      <td>-4.548</td>
      <td>0.563510</td>
      <td>0.685227</td>
      <td>0.375</td>
      <td>0.293</td>
      <td>0.357</td>
      <td>0.869</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5.0</td>
      <td>0.782854</td>
      <td>0.090648</td>
      <td>117.977</td>
      <td>0.035536</td>
      <td>0.000227</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>269.37574</td>
      <td>-4.256</td>
      <td>0.377586</td>
      <td>0.490146</td>
      <td>0.649</td>
      <td>0.628</td>
      <td>0.724</td>
      <td>0.724</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>10</th>
      <td>8.0</td>
      <td>0.511694</td>
      <td>0.159764</td>
      <td>86.350</td>
      <td>0.187572</td>
      <td>0.037796</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>174.13333</td>
      <td>-7.871</td>
      <td>0.539036</td>
      <td>0.792948</td>
      <td>0.212</td>
      <td>0.521</td>
      <td>0.263</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>0.599368</td>
      <td>0.116890</td>
      <td>151.496</td>
      <td>0.035372</td>
      <td>0.175869</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>309.34667</td>
      <td>-8.742</td>
      <td>0.249932</td>
      <td>0.443485</td>
      <td>0.566</td>
      <td>0.608</td>
      <td>0.217</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>12</th>
      <td>11.0</td>
      <td>0.324839</td>
      <td>0.325669</td>
      <td>121.379</td>
      <td>0.655687</td>
      <td>0.030058</td>
      <td>0.000032</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>208.57333</td>
      <td>-17.942</td>
      <td>0.752856</td>
      <td>0.604906</td>
      <td>0.525</td>
      <td>0.628</td>
      <td>0.429</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5.0</td>
      <td>0.816507</td>
      <td>0.212694</td>
      <td>129.975</td>
      <td>0.110272</td>
      <td>0.000264</td>
      <td>0.000003</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>226.85333</td>
      <td>-6.638</td>
      <td>0.671683</td>
      <td>0.590436</td>
      <td>0.023</td>
      <td>0.273</td>
      <td>0.680</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>14</th>
      <td>6.0</td>
      <td>0.831990</td>
      <td>0.464863</td>
      <td>170.237</td>
      <td>0.392644</td>
      <td>0.040739</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>210.31937</td>
      <td>-4.617</td>
      <td>0.573154</td>
      <td>0.609290</td>
      <td>0.431</td>
      <td>0.553</td>
      <td>0.095</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>15</th>
      <td>9.0</td>
      <td>0.908129</td>
      <td>0.180950</td>
      <td>171.135</td>
      <td>0.044720</td>
      <td>0.040514</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>359.54667</td>
      <td>-5.997</td>
      <td>0.518113</td>
      <td>0.354004</td>
      <td>0.890</td>
      <td>0.638</td>
      <td>0.886</td>
      <td>0.911</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.0</td>
      <td>0.906022</td>
      <td>0.341704</td>
      <td>145.048</td>
      <td>0.324825</td>
      <td>0.070587</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>229.62667</td>
      <td>-1.641</td>
      <td>0.394620</td>
      <td>0.666617</td>
      <td>0.348</td>
      <td>0.353</td>
      <td>0.107</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.0</td>
      <td>0.446286</td>
      <td>0.499771</td>
      <td>104.044</td>
      <td>0.143111</td>
      <td>0.006503</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>311.98667</td>
      <td>-17.673</td>
      <td>0.482862</td>
      <td>0.602205</td>
      <td>0.161</td>
      <td>0.153</td>
      <td>0.551</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>0.518301</td>
      <td>0.089267</td>
      <td>117.364</td>
      <td>0.349597</td>
      <td>0.028045</td>
      <td>0.000021</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>277.37279</td>
      <td>-10.366</td>
      <td>0.297294</td>
      <td>0.489609</td>
      <td>0.234</td>
      <td>0.548</td>
      <td>0.530</td>
      <td>0.563</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>19</th>
      <td>8.0</td>
      <td>0.643952</td>
      <td>0.121978</td>
      <td>124.951</td>
      <td>0.034232</td>
      <td>0.011873</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>254.85333</td>
      <td>-6.494</td>
      <td>0.486868</td>
      <td>0.671058</td>
      <td>0.048</td>
      <td>0.044</td>
      <td>0.347</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10.0</td>
      <td>0.674971</td>
      <td>0.060643</td>
      <td>140.043</td>
      <td>0.212844</td>
      <td>0.050310</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>250.66667</td>
      <td>-3.368</td>
      <td>0.559535</td>
      <td>0.734192</td>
      <td>0.433</td>
      <td>0.469</td>
      <td>0.595</td>
      <td>0.070</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>21</th>
      <td>11.0</td>
      <td>0.953475</td>
      <td>0.188341</td>
      <td>100.261</td>
      <td>0.076731</td>
      <td>0.014586</td>
      <td>0.017329</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>405.50000</td>
      <td>-8.760</td>
      <td>0.439079</td>
      <td>0.273381</td>
      <td>0.862</td>
      <td>0.687</td>
      <td>0.464</td>
      <td>0.948</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3.0</td>
      <td>0.828215</td>
      <td>0.131678</td>
      <td>114.725</td>
      <td>0.069248</td>
      <td>0.166942</td>
      <td>0.743609</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>169.41333</td>
      <td>-6.273</td>
      <td>0.761261</td>
      <td>0.742523</td>
      <td>0.000</td>
      <td>0.149</td>
      <td>0.789</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2.0</td>
      <td>0.597694</td>
      <td>0.130781</td>
      <td>109.921</td>
      <td>0.027832</td>
      <td>0.000206</td>
      <td>0.497340</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>295.05333</td>
      <td>-6.449</td>
      <td>0.131938</td>
      <td>0.514961</td>
      <td>0.319</td>
      <td>0.419</td>
      <td>0.491</td>
      <td>0.998</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.0</td>
      <td>0.790401</td>
      <td>0.226377</td>
      <td>142.154</td>
      <td>0.054368</td>
      <td>0.102046</td>
      <td>0.000016</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>299.60000</td>
      <td>-3.273</td>
      <td>0.235483</td>
      <td>0.531062</td>
      <td>0.267</td>
      <td>0.398</td>
      <td>0.294</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>25</th>
      <td>8.0</td>
      <td>0.956780</td>
      <td>0.078926</td>
      <td>134.992</td>
      <td>0.074067</td>
      <td>0.043111</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>178.01333</td>
      <td>-2.336</td>
      <td>0.681865</td>
      <td>0.639745</td>
      <td>0.552</td>
      <td>0.591</td>
      <td>0.916</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2.0</td>
      <td>0.850453</td>
      <td>0.192271</td>
      <td>114.011</td>
      <td>0.068293</td>
      <td>0.080688</td>
      <td>0.000003</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>287.08000</td>
      <td>-1.372</td>
      <td>0.641048</td>
      <td>0.932810</td>
      <td>0.768</td>
      <td>0.468</td>
      <td>0.914</td>
      <td>0.963</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.0</td>
      <td>0.352378</td>
      <td>0.105419</td>
      <td>142.974</td>
      <td>0.193916</td>
      <td>0.099264</td>
      <td>0.004144</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>272.19592</td>
      <td>-13.982</td>
      <td>0.180567</td>
      <td>0.712630</td>
      <td>0.542</td>
      <td>0.426</td>
      <td>0.439</td>
      <td>0.982</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>28</th>
      <td>5.0</td>
      <td>0.361470</td>
      <td>0.110629</td>
      <td>146.387</td>
      <td>0.024319</td>
      <td>0.711393</td>
      <td>0.000015</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>309.16000</td>
      <td>-10.599</td>
      <td>0.227419</td>
      <td>0.567821</td>
      <td>0.533</td>
      <td>0.576</td>
      <td>0.311</td>
      <td>0.990</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>29</th>
      <td>8.0</td>
      <td>0.737760</td>
      <td>0.101605</td>
      <td>118.002</td>
      <td>0.034670</td>
      <td>0.391096</td>
      <td>0.000043</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>205.02667</td>
      <td>-6.090</td>
      <td>0.525055</td>
      <td>0.593236</td>
      <td>0.542</td>
      <td>0.770</td>
      <td>0.720</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8859</th>
      <td>9.0</td>
      <td>0.529422</td>
      <td>0.591604</td>
      <td>147.949</td>
      <td>0.094531</td>
      <td>0.063366</td>
      <td>0.108790</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>392.28299</td>
      <td>-12.335</td>
      <td>0.901001</td>
      <td>0.741657</td>
      <td>0.238</td>
      <td>0.323</td>
      <td>0.769</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8860</th>
      <td>4.0</td>
      <td>0.697638</td>
      <td>0.303938</td>
      <td>92.671</td>
      <td>0.348999</td>
      <td>0.004146</td>
      <td>0.747203</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>74.09293</td>
      <td>-6.693</td>
      <td>0.694312</td>
      <td>0.612750</td>
      <td>0.000</td>
      <td>0.083</td>
      <td>0.366</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>8861</th>
      <td>2.0</td>
      <td>0.888037</td>
      <td>0.344913</td>
      <td>93.406</td>
      <td>0.157239</td>
      <td>0.626277</td>
      <td>0.000006</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>147.21333</td>
      <td>-6.550</td>
      <td>0.790581</td>
      <td>0.538191</td>
      <td>0.474</td>
      <td>0.588</td>
      <td>0.116</td>
      <td>0.615</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8862</th>
      <td>2.0</td>
      <td>0.679585</td>
      <td>0.202416</td>
      <td>103.983</td>
      <td>0.032782</td>
      <td>0.133261</td>
      <td>0.078730</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>241.16000</td>
      <td>-14.639</td>
      <td>0.706314</td>
      <td>0.414220</td>
      <td>0.708</td>
      <td>0.735</td>
      <td>0.641</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8863</th>
      <td>4.0</td>
      <td>0.789400</td>
      <td>0.076088</td>
      <td>88.675</td>
      <td>0.088033</td>
      <td>0.351537</td>
      <td>0.874648</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>554.44000</td>
      <td>-7.474</td>
      <td>0.316857</td>
      <td>0.290259</td>
      <td>0.932</td>
      <td>0.706</td>
      <td>0.180</td>
      <td>0.365</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8864</th>
      <td>0.0</td>
      <td>0.486410</td>
      <td>0.152350</td>
      <td>145.483</td>
      <td>0.029142</td>
      <td>0.675752</td>
      <td>0.971118</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>164.64000</td>
      <td>-12.737</td>
      <td>0.872131</td>
      <td>0.606663</td>
      <td>0.972</td>
      <td>0.280</td>
      <td>0.912</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8865</th>
      <td>0.0</td>
      <td>0.572654</td>
      <td>0.258702</td>
      <td>95.956</td>
      <td>0.032420</td>
      <td>0.751582</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>371.26667</td>
      <td>-10.632</td>
      <td>0.544541</td>
      <td>0.469454</td>
      <td>0.833</td>
      <td>0.969</td>
      <td>0.554</td>
      <td>0.924</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8866</th>
      <td>9.0</td>
      <td>0.776514</td>
      <td>0.080175</td>
      <td>117.600</td>
      <td>0.041781</td>
      <td>0.522408</td>
      <td>0.000001</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>134.23456</td>
      <td>-4.496</td>
      <td>0.832975</td>
      <td>0.667857</td>
      <td>0.613</td>
      <td>0.613</td>
      <td>0.731</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8867</th>
      <td>4.0</td>
      <td>0.775220</td>
      <td>0.242049</td>
      <td>171.416</td>
      <td>0.059662</td>
      <td>0.482566</td>
      <td>0.055283</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>140.93333</td>
      <td>-10.406</td>
      <td>0.724443</td>
      <td>0.445160</td>
      <td>0.668</td>
      <td>0.554</td>
      <td>0.087</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8868</th>
      <td>4.0</td>
      <td>0.765977</td>
      <td>0.270061</td>
      <td>77.096</td>
      <td>0.101700</td>
      <td>0.493516</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>181.30621</td>
      <td>-5.938</td>
      <td>0.889122</td>
      <td>0.564186</td>
      <td>0.303</td>
      <td>0.254</td>
      <td>0.285</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8869</th>
      <td>0.0</td>
      <td>0.976064</td>
      <td>0.300835</td>
      <td>114.989</td>
      <td>0.130319</td>
      <td>0.122308</td>
      <td>0.002517</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>226.97333</td>
      <td>-2.854</td>
      <td>0.171776</td>
      <td>0.249475</td>
      <td>0.784</td>
      <td>0.618</td>
      <td>0.502</td>
      <td>0.851</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8870</th>
      <td>4.0</td>
      <td>0.901357</td>
      <td>0.306848</td>
      <td>131.742</td>
      <td>0.054015</td>
      <td>0.039949</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>143.50667</td>
      <td>-7.272</td>
      <td>0.454201</td>
      <td>0.510201</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.141</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8871</th>
      <td>3.0</td>
      <td>0.369164</td>
      <td>0.122126</td>
      <td>118.544</td>
      <td>0.036121</td>
      <td>0.880627</td>
      <td>0.219668</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>183.01333</td>
      <td>-6.116</td>
      <td>0.499139</td>
      <td>0.310018</td>
      <td>0.406</td>
      <td>0.558</td>
      <td>0.055</td>
      <td>0.527</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8872</th>
      <td>7.0</td>
      <td>0.773904</td>
      <td>0.095558</td>
      <td>77.610</td>
      <td>0.045200</td>
      <td>0.000005</td>
      <td>0.057821</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>141.73415</td>
      <td>-9.914</td>
      <td>0.543488</td>
      <td>0.223012</td>
      <td>0.678</td>
      <td>0.596</td>
      <td>0.388</td>
      <td>0.868</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8873</th>
      <td>0.0</td>
      <td>0.801679</td>
      <td>0.212999</td>
      <td>126.031</td>
      <td>0.070304</td>
      <td>0.224264</td>
      <td>0.875990</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>135.37333</td>
      <td>-6.966</td>
      <td>0.618488</td>
      <td>0.658323</td>
      <td>0.449</td>
      <td>0.476</td>
      <td>0.852</td>
      <td>0.723</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8874</th>
      <td>10.0</td>
      <td>0.483210</td>
      <td>0.099431</td>
      <td>97.232</td>
      <td>0.034740</td>
      <td>0.760556</td>
      <td>0.885722</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>343.32000</td>
      <td>-12.737</td>
      <td>0.270693</td>
      <td>0.228276</td>
      <td>0.865</td>
      <td>0.496</td>
      <td>0.108</td>
      <td>0.840</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8875</th>
      <td>2.0</td>
      <td>0.515708</td>
      <td>0.117357</td>
      <td>96.013</td>
      <td>0.025252</td>
      <td>0.122950</td>
      <td>0.000270</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>325.14667</td>
      <td>-8.295</td>
      <td>0.398064</td>
      <td>0.550200</td>
      <td>0.645</td>
      <td>0.685</td>
      <td>0.377</td>
      <td>0.932</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8876</th>
      <td>1.0</td>
      <td>0.944113</td>
      <td>0.175283</td>
      <td>173.052</td>
      <td>0.079482</td>
      <td>0.211674</td>
      <td>0.001535</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>207.86667</td>
      <td>-7.956</td>
      <td>0.376798</td>
      <td>0.391476</td>
      <td>0.816</td>
      <td>0.434</td>
      <td>0.745</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8877</th>
      <td>5.0</td>
      <td>0.713960</td>
      <td>0.164678</td>
      <td>110.671</td>
      <td>0.390050</td>
      <td>0.296917</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>267.26667</td>
      <td>-5.602</td>
      <td>0.775812</td>
      <td>0.572286</td>
      <td>0.342</td>
      <td>0.388</td>
      <td>0.059</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>8878</th>
      <td>0.0</td>
      <td>0.667943</td>
      <td>0.587038</td>
      <td>96.934</td>
      <td>0.042427</td>
      <td>0.522196</td>
      <td>0.500257</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>491.25333</td>
      <td>-12.230</td>
      <td>0.925612</td>
      <td>0.523138</td>
      <td>0.697</td>
      <td>0.624</td>
      <td>0.590</td>
      <td>0.992</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8879</th>
      <td>4.0</td>
      <td>0.827205</td>
      <td>0.476990</td>
      <td>116.396</td>
      <td>0.083865</td>
      <td>0.000154</td>
      <td>0.814663</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>531.24000</td>
      <td>-7.440</td>
      <td>0.325596</td>
      <td>0.374048</td>
      <td>0.402</td>
      <td>0.237</td>
      <td>0.129</td>
      <td>0.204</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8880</th>
      <td>10.0</td>
      <td>0.494515</td>
      <td>0.111540</td>
      <td>124.867</td>
      <td>0.037051</td>
      <td>0.192127</td>
      <td>0.463980</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>236.06667</td>
      <td>-9.888</td>
      <td>0.608016</td>
      <td>0.579593</td>
      <td>0.570</td>
      <td>0.388</td>
      <td>0.433</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8881</th>
      <td>3.0</td>
      <td>0.558495</td>
      <td>0.325305</td>
      <td>172.045</td>
      <td>0.078690</td>
      <td>0.904698</td>
      <td>0.000013</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>132.74998</td>
      <td>-15.024</td>
      <td>0.662363</td>
      <td>0.333256</td>
      <td>0.423</td>
      <td>0.400</td>
      <td>0.300</td>
      <td>0.273</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8882</th>
      <td>7.0</td>
      <td>0.368245</td>
      <td>0.574043</td>
      <td>124.537</td>
      <td>0.044899</td>
      <td>0.308824</td>
      <td>0.740456</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1229.58322</td>
      <td>-19.000</td>
      <td>0.512359</td>
      <td>0.438267</td>
      <td>0.207</td>
      <td>0.542</td>
      <td>0.327</td>
      <td>0.426</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8883</th>
      <td>7.0</td>
      <td>0.948452</td>
      <td>0.244831</td>
      <td>168.472</td>
      <td>0.057166</td>
      <td>0.011390</td>
      <td>0.000224</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>172.40000</td>
      <td>-2.189</td>
      <td>0.479637</td>
      <td>0.096399</td>
      <td>0.931</td>
      <td>0.663</td>
      <td>0.048</td>
      <td>0.632</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8884</th>
      <td>1.0</td>
      <td>0.726530</td>
      <td>0.241541</td>
      <td>91.764</td>
      <td>0.329669</td>
      <td>0.055165</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>88.94667</td>
      <td>-7.469</td>
      <td>0.553143</td>
      <td>0.779995</td>
      <td>0.666</td>
      <td>0.729</td>
      <td>0.561</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>8885</th>
      <td>10.0</td>
      <td>0.622512</td>
      <td>0.202207</td>
      <td>105.176</td>
      <td>0.034968</td>
      <td>0.900876</td>
      <td>0.845736</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>222.10667</td>
      <td>-7.497</td>
      <td>0.413890</td>
      <td>0.310395</td>
      <td>0.419</td>
      <td>0.656</td>
      <td>0.520</td>
      <td>0.986</td>
      <td>rock</td>
    </tr>
    <tr>
      <th>8886</th>
      <td>4.0</td>
      <td>0.622215</td>
      <td>0.080331</td>
      <td>90.263</td>
      <td>0.027364</td>
      <td>0.283297</td>
      <td>0.536592</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>335.77850</td>
      <td>-14.571</td>
      <td>0.723777</td>
      <td>0.495290</td>
      <td>0.451</td>
      <td>0.390</td>
      <td>0.906</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>8887</th>
      <td>8.0</td>
      <td>0.498902</td>
      <td>0.399844</td>
      <td>90.038</td>
      <td>0.105419</td>
      <td>0.158190</td>
      <td>0.776264</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>576.93333</td>
      <td>-11.267</td>
      <td>0.286561</td>
      <td>0.689493</td>
      <td>0.417</td>
      <td>0.415</td>
      <td>0.291</td>
      <td>1.000</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>8888</th>
      <td>5.0</td>
      <td>0.523364</td>
      <td>0.109184</td>
      <td>127.986</td>
      <td>0.030111</td>
      <td>0.005962</td>
      <td>0.812327</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>237.68000</td>
      <td>-11.519</td>
      <td>0.616658</td>
      <td>0.551328</td>
      <td>0.771</td>
      <td>0.502</td>
      <td>0.895</td>
      <td>1.000</td>
      <td>rock</td>
    </tr>
  </tbody>
</table>
<p>8889 rows × 18 columns</p>
</div>




```python
label_genres = np.array(rock_rap['genres'])
final_features = rock_rap.drop('genres',axis = 1).astype(float)
final_features.isnull().any()
```




    key                          False
    energy                       False
    liveliness                   False
    tempo                        False
    speechiness                   True
    acousticness                 False
    instrumentalness             False
    time_signature               False
    duration                     False
    loudness                     False
    valence                      False
    danceability                 False
    mode                         False
    time_signature_confidence    False
    tempo_confidence             False
    key_confidence               False
    mode_confidence              False
    dtype: bool




```python
final_features = final_features.fillna(final_features.median())
final_features.isnull().any()
```




    key                          False
    energy                       False
    liveliness                   False
    tempo                        False
    speechiness                  False
    acousticness                 False
    instrumentalness             False
    time_signature               False
    duration                     False
    loudness                     False
    valence                      False
    danceability                 False
    mode                         False
    time_signature_confidence    False
    tempo_confidence             False
    key_confidence               False
    mode_confidence              False
    dtype: bool




```python
knn = neighbors.KNeighborsClassifier(n_neighbors = 3)
standard_scaler = preprocessing.StandardScaler()
final_features = standard_scaler.fit_transform(final_features)
```


```python
X_train,X_test,y_train,y_test = cross_validation.train_test_split(final_features,label_genres,test_size = 0.2,random_state = 101)
```


```python
knn.fit(X_train,y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=3, p=2,
               weights='uniform')




```python
pred = knn.predict(X_test)
```


```python
from nltk import ConfusionMatrix
print(ConfusionMatrix(list(y_test), list(pred)))
```

         |         r |
         |    r    o |
         |    a    c |
         |    p    k |
    -----+-----------+
     rap | <392> 102 |
    rock |   45<1239>|
    -----+-----------+
    (row = reference; col = test)
    



```python
print(metrics.classification_report(y_test,pred))
```

                 precision    recall  f1-score   support
    
            rap       0.90      0.79      0.84       494
           rock       0.92      0.96      0.94      1284
    
    avg / total       0.92      0.92      0.92      1778
    



```python
scores = cross_validation.cross_val_score(knn,final_features,label_genres,cv=10,scoring = 'accuracy')
print(scores.mean())
```

    0.919334113186



```python
print(metrics.accuracy_score(y_test,pred))
```

    0.922947131609


## here I want to find the optimal K for for the highest mean scores:



```python
def optimal_k(X,y,cv=10,scoring='accuracy'):
    k_list = list(range(1,11))

    # subsetting just the odd ones
    n = [k for k in k_list]

    # empty list that will hold cv scores
    cv_scores = []

    # perform 10-fold cross validation
    for k in n:
        knn = neighbors.KNeighborsClassifier(n_neighbors = k)
        scores = cross_validation.cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
        MSE = [1 - x for x in cv_scores]

    # determining best k
    optimal_k = n[MSE.index(min(MSE))]
    return cv_scores[optimal_k]
```


```python
print(optimal_k(final_features,label_genres,cv=10,scoring='accuracy')) ##this yields with k =8
```

    0.91944482826


# With balanced data


```python
from collections import Counter
Counter(rock_rap['genres'])
```




    Counter({'rock': 6437, 'rap': 2452})




```python
scores = []
standard_scaler = preprocessing.StandardScaler()
for i in range(10): #this yields 10 random samples
    rock_2000 = rock_rap.loc[rock_rap['genres'] == 'rock'].sample(n=2000)
    rap_2000 = rock_rap.loc[rock_rap['genres'] == 'rap'].sample(n=2000)
    df_2000 = pd.concat([rock_2000,rap_2000], ignore_index=True)
    df_2000 = df_2000.fillna(df_2000.median(),inplace=True)
    knn = neighbors.KNeighborsClassifier(n_neighbors = 6)
    
    label_2000 = np.array(df_2000['genres'])
    final_2000 = df_2000.drop('genres',axis = 1).astype(float)
    final_2000 = standard_scaler.fit_transform(final_2000)
    X_train,X_test,y_train,y_test = cross_validation.train_test_split(final_features,label_genres,test_size = 0.2,random_state = 101)
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, pred)
    scores.append(accuracy)
```


```python
print(scores)
```

    [0.92294713160854891, 0.92294713160854891, 0.92294713160854891, 0.92294713160854891, 0.92294713160854891, 0.92294713160854891, 0.92294713160854891, 0.92294713160854891, 0.92294713160854891, 0.92294713160854891]


## --> so bascially with unbalanced or balanced data, the accuracy scores aren't much different

## I try to with cross_val_score by using the function optimal_k but encounter this error....


```python
scores = []
standard_scaler = preprocessing.StandardScaler()
for i in range(10):
    rock_2000 = rock_rap.loc[rock_rap['genres'] == 'rock'].sample(n=2000)
    rap_2000 = rock_rap.loc[rock_rap['genres'] == 'rap'].sample(n=2000)
    df_2000 = pd.concat([rock_2000,rap_2000], ignore_index=True)
    df_2000 = df_2000.fillna(df_2000.median(),inplace=True)
    
    label_2000 = np.array(df_2000['genres'])
    final_2000 = df_2000.drop('genres',axis = 1).astype(float)
    final_2000 = standard_scaler.fit_transform(final_2000)
    scores.append(optimal_k(final_2000,label_2000,cv=10,scoring='accuracy'))
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-406-316cd807ae01> in <module>()
         10     final_2000 = df_2000.drop('genres',axis = 1).astype(float)
         11     final_2000 = standard_scaler.fit_transform(final_2000)
    ---> 12     scores.append(optimal_k(final_2000,label_2000,cv=10,scoring='accuracy'))
    

    <ipython-input-404-bf998b74b12a> in optimal_k(X, y, cv, scoring)
         17     # determining best k
         18     optimal_k = n[MSE.index(min(MSE))]
    ---> 19     return cv_scores[optimal_k]
    

    IndexError: list index out of range

