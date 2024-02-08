import galai as gal
# from galai.notebook_utils import *
model = gal.load_model("standard", num_gpus = 1, parallelize = True)
# print(model.generate("The Transformer architecture [START_REF]"))

prompt = """ Imagine you are a research scientist Analyse the following paper and generate 5 possible future research ideas from it:  
introduction. Numerous epidemiological studies have been conducted on the effects of particulate matter (PM) with a diameter of less than 2.5 µm (PM2.5) on human health (Dockery et al. 1993; Hart et al. 2015; Li et al. 2018). The literature, consequently,
is replete with evidence of its negative effects on human beings (Burgan et al. 2010; Cesaroni et al. 2014; Atkinson et al. 2014; Yuan et al. 2019).
However, during recent years, general public concern— and interest—regarding ultrafine particles (UFPs), which are PM with a diameter of less than 100 nm, has also increased. The characteristics of PM depend on its size and particle origin (Mühlfeld et al. 2008; Morawska et al. 2008), and there are four distinguishing characteristics of UFPs. First, they constitute less than 20% of the total mass concentration of particles, but more than 90% of the total number concentration of particles, compared to PM with a diameter of less than 10 µm (PM10) and PM2.5 (Kittelson 1998; Kumar et al. 2009). Second, UFPs have a high share in direct emissions from anthropogenic sources, such as road transportation and power plants, whereas PM2.5 has a high share in secondary sources, that is, through chemical processes in the atmosphere (Kittelson 1998; Morawska et al. 2008; Liang et al. 2016). UFPs emitted from road transportation Responsible Editor: Philippe Garrigues * Youngsang Cho y.cho@yonsei.ac.kr Eunjung Cho ejung720@yonsei.ac.kr 1 Department of Industrial Engineering, College of Engineering, Yonsei University, 50 Yonsei-Ro, Seodaemun-Gu, Seoul 03722, South Korea
2 Technical Analysis Center, National Institute of Green Technology, 173, Toegye-Ro, Jung-Gu, Seoul 04554, South Korea

5 possible future research ideas from this paper are: """

def inference(text):
    return model.generate(prompt, new_doc=True, max_new_tokens=500)

def main():
    print(inference(prompt))

if __name__ == '__main__':
    main()