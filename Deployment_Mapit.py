from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import mapit

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', hasil=0)

@app.route('/', methods=['POST'])
def predict():
    
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''

    df = pd.read_csv('MainData.csv',sep=';')

    #Resamling Demography Criteria
    df["jumlah_kepadatan_(jiwa/km2)"] = (MinMaxScaler(feature_range = (-1,1)).fit_transform(df.iloc[:,6].values.reshape(-1,1))*-1)
    df["Economic_status"] = (MinMaxScaler(feature_range = (-1,1)).fit_transform(df.iloc[:,13].values.reshape(-1,1)))
    df["persentase_umur_15_34"] = (MinMaxScaler(feature_range = (-1,1)).fit_transform(df.iloc[:,14].values.reshape(-1,1))*-1)
    df["laki_laki"] = (MinMaxScaler(feature_range = (-1,1)).fit_transform(df.iloc[:,2].values.reshape(-1,1)))
    df["perempuan"] = (MinMaxScaler(feature_range = (-1,1)).fit_transform(df.iloc[:,3].values.reshape(-1,1))*-1)

    df.drop(['jumlah_penduduk','jumlah_luas_wilayah_(km2)','umur_15-19','umur_20-24','umur_25-29','umur_30-34','umur_35-39','umur_40-44'], axis=1,inplace=True)

    kriteria = pd.DataFrame(index=['Economic_Status', 'Kepadatan', 'Persentase_umur_15_34','Laki', 'Perempuan', 'total'], columns=['Economic_Status','Kepadatan','Persentase_umur_15_34','Laki','Perempuan'])

    ES = int(request.form['status_ekonomi'])
    Kp = int(request.form['kepadatan_penduduk'])
    PU = int(request.form['persentase_umur'])
    L = int(request.form['jumlah_laki'])
    P = int(request.form['jumlah_perempuan'])
    # ES = status_ekonomi
    # Kp = kepadatan_penduduk
    # PU = persentase_umur
    # L = jumlah_laki
    # P = jumlah_perempuan

    def compare(a, b):
        if a>b:
            c = 1/(a-b)
        elif b>a :
            c = b-a
        else:
            c = 1
        return c

    kriteria.Economic_Status[0] = compare(ES, ES)
    kriteria.Economic_Status[1] = compare(ES, Kp)
    kriteria.Economic_Status[2] = compare(ES, PU)
    kriteria.Economic_Status[3] = compare(ES, L)
    kriteria.Economic_Status[4] = compare(ES, P)
    kriteria.Economic_Status[5] = kriteria.Economic_Status.sum()

    kriteria.Kepadatan[0] = compare(Kp, ES)
    kriteria.Kepadatan[1] = compare(Kp, Kp)
    kriteria.Kepadatan[2] = compare(Kp, PU)
    kriteria.Kepadatan[3] = compare(Kp, L)
    kriteria.Kepadatan[4] = compare(Kp, P)
    kriteria.Kepadatan[5] = kriteria.Kepadatan.sum()

    kriteria.Persentase_umur_15_34[0] = compare(PU, ES)
    kriteria.Persentase_umur_15_34[1] = compare(PU, Kp)
    kriteria.Persentase_umur_15_34[2] = compare(PU, PU)
    kriteria.Persentase_umur_15_34[3] = compare(PU, L)
    kriteria.Persentase_umur_15_34[4] = compare(PU, P)
    kriteria.Persentase_umur_15_34[5] = kriteria.Persentase_umur_15_34.sum()

    kriteria.Laki[0] = compare(L, ES)
    kriteria.Laki[1] = compare(L, Kp)
    kriteria.Laki[2] = compare(L, PU)
    kriteria.Laki[3] = compare(L, L)
    kriteria.Laki[4] = compare(L, P)
    kriteria.Laki[5] = kriteria.Laki.sum()

    kriteria.Perempuan[0] = compare(L, ES)
    kriteria.Perempuan[1] = compare(L, Kp)
    kriteria.Perempuan[2] = compare(L, PU)
    kriteria.Perempuan[3] = compare(L, L)
    kriteria.Perempuan[4] = compare(L, P)
    kriteria.Perempuan[5] = kriteria.Perempuan.sum()

    #SCORE DI TIAP PERBANDINGAN DIBAGI DENGAN TOTAL ES

    kriteria.Economic_Status[0] = kriteria.Economic_Status[0] / kriteria.Economic_Status[5]
    kriteria.Economic_Status[1] = kriteria.Economic_Status[1] / kriteria.Economic_Status[5]
    kriteria.Economic_Status[2] = kriteria.Economic_Status[2] / kriteria.Economic_Status[5]
    kriteria.Economic_Status[3] = kriteria.Economic_Status[3] / kriteria.Economic_Status[5]
    kriteria.Economic_Status[4] = kriteria.Economic_Status[4] / kriteria.Economic_Status[5]

    #SCORE DI TIAP PERBANDINGAN DIBAGI DENGAN TOTAL Kp

    kriteria.Kepadatan[0] = kriteria.Kepadatan[0] / kriteria.Kepadatan[5]
    kriteria.Kepadatan[1] = kriteria.Kepadatan[1] / kriteria.Kepadatan[5]
    kriteria.Kepadatan[2] = kriteria.Kepadatan[2] / kriteria.Kepadatan[5]
    kriteria.Kepadatan[3] = kriteria.Kepadatan[3] / kriteria.Kepadatan[5]
    kriteria.Kepadatan[4] = kriteria.Kepadatan[4] / kriteria.Kepadatan[5]

    #SCORE DI TIAP PERBANDINGAN DIBAGI DENGAN TOTAL PU

    kriteria.Persentase_umur_15_34[0] = kriteria.Persentase_umur_15_34[0] / kriteria.Persentase_umur_15_34[5]
    kriteria.Persentase_umur_15_34[1] = kriteria.Persentase_umur_15_34[1] / kriteria.Persentase_umur_15_34[5]
    kriteria.Persentase_umur_15_34[2] = kriteria.Persentase_umur_15_34[2] / kriteria.Persentase_umur_15_34[5]
    kriteria.Persentase_umur_15_34[3] = kriteria.Persentase_umur_15_34[3] / kriteria.Persentase_umur_15_34[5]
    kriteria.Persentase_umur_15_34[4] = kriteria.Persentase_umur_15_34[4] / kriteria.Persentase_umur_15_34[5]

    #SCORE DI TIAP PERBANDINGAN DIBAGI DENGAN TOTAL L

    kriteria.Laki[0] = kriteria.Laki[0] / kriteria.Laki[5]
    kriteria.Laki[1] = kriteria.Laki[1] / kriteria.Laki[5]
    kriteria.Laki[2] = kriteria.Laki[2] / kriteria.Laki[5]
    kriteria.Laki[3] = kriteria.Laki[3] / kriteria.Laki[5]
    kriteria.Laki[4] = kriteria.Laki[4] / kriteria.Laki[5]

    #SCORE DI TIAP PERBANDINGAN DIBAGI DENGAN TOTAL P

    kriteria.Perempuan[0] = kriteria.Perempuan[0] / kriteria.Perempuan[5]
    kriteria.Perempuan[1] = kriteria.Perempuan[1] / kriteria.Perempuan[5]
    kriteria.Perempuan[2] = kriteria.Perempuan[2] / kriteria.Perempuan[5]
    kriteria.Perempuan[3] = kriteria.Perempuan[3] / kriteria.Perempuan[5]
    kriteria.Perempuan[4] = kriteria.Perempuan[4] / kriteria.Perempuan[5]

    totalp_ES = kriteria.Economic_Status[0]+kriteria.Kepadatan[0]+kriteria.Persentase_umur_15_34[0]+kriteria.Laki[0]+kriteria.Perempuan[0]
    totalp_Kp = kriteria.Economic_Status[1]+kriteria.Kepadatan[1]+kriteria.Persentase_umur_15_34[1]+kriteria.Laki[1]+kriteria.Perempuan[1]
    totalp_PU = kriteria.Economic_Status[2]+kriteria.Kepadatan[2]+kriteria.Persentase_umur_15_34[2]+kriteria.Laki[2]+kriteria.Perempuan[2]
    totalp_L = kriteria.Economic_Status[3]+kriteria.Kepadatan[3]+kriteria.Persentase_umur_15_34[3]+kriteria.Laki[3]+kriteria.Perempuan[3]
    totalp_P = kriteria.Economic_Status[4]+kriteria.Kepadatan[4]+kriteria.Persentase_umur_15_34[4]+kriteria.Laki[4]+kriteria.Perempuan[4]

    E_param = [totalp_ES,totalp_Kp,totalp_PU,totalp_L,totalp_P]
    E_param

    ES_weight = E_param[0] / 5
    Kp_weight = E_param[1] / 5
    PU_weight = E_param[2] / 5
    L_weight = E_param[3] / 5
    P_weight = E_param[4] / 5

    df_Weight = [ES_weight, Kp_weight, PU_weight, L_weight, P_weight ]

    #AHP for Demography
    df["df_Total"] = ((df["Economic_status"]*df_Weight[0])+
                            (df["jumlah_kepadatan_(jiwa/km2)"]*df_Weight[1])+
                            (df["persentase_umur_15_34"]*df_Weight[2])+
                            (df["laki_laki"]*df_Weight[3])+
                            (df["perempuan"]*df_Weight[4]))

    df["df_Total"] = (MinMaxScaler(feature_range = (-1,1)).fit_transform(df.iloc[:,7].values.reshape(-1,1)))
    score=df[['df_Total','nomor_kelurahan']]

    df.sort_values(by='df_Total',ascending=False,ignore_index=True,inplace=True)

    df.pop('Unnamed: 2')
    df.pop('perempuan')
    df.pop('jumlah_kepadatan_(jiwa/km2)')
    df.pop('persentase_umur_15_34')
    df.pop('Economic_status')
    df.pop('laki_laki')
    df.pop('df_Total')

    graphJSON = mapit.showgeo(score)

    print(df.head())
    print(score.head())
    print(ES, Kp, PU, L, P)

    return render_template('index.html', 
        tables=[df.head(10).to_html()], 
        titles=df.columns.values,
        graph_json = graphJSON,
    )

if __name__ == '__main__':
    app.run(debug=True)