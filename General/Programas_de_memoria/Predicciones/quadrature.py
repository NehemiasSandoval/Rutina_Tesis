import numpy as np

def quadratureRule_fine(K):
    """
    Fine quadrature rule, to solve the near singular integrals.
    Arguments
    ----------
    K: int (1, 7, 13, 17, 19, 25, 37, 48, 52, 61, 79), number of Gauss points
       per element.
    Returns
    --------
    X: array, position of the gauss quadrature points.
    W: array, gauss quadrature weights.
    """
    # yapf: disable
    # 1 Gauss point
    print("Se usaran" + str(K) + "Puntos de cuadratura gaussiana")
    if K==1:
        X = np.array([1./3., 1./3., 1./3.])
        W = np.array([1.])

    # 7 Gauss points
    if K==7:
        a = 1./3.
        b1 = 0.059715871789770; b2 = 0.470142064105115
        c1 = 0.797426985353087; c2 = 0.101286507323456

        wa = 0.225000000000000
        wb = 0.132394152788506
        wc = 0.125939180544827

        X = np.array([a,a,a,
                       b1,b2,b2,b2,b1,b2,b2,b2,b1,
                       c1,c2,c2,c2,c1,c2,c2,c2,c1])
        W = np.array([wa,wb,wb,wb,wc,wc,wc])

    # 13 Gauss points
    if K==13:
        a = 1./3.
        b1 = 0.479308067841920; b2 = 0.260345966079040
        c1 = 0.869739794195568; c2 = 0.065130102902216
        d1 = 0.048690315425316; d2 = 0.312865496004874; d3 = 0.638444188569810
        wa = -0.149570044467682
        wb = 0.175615257433208
        wc = 0.053347235608838
        wd = 0.077113760890257

        X = np.array([a,a,a,
                       b1,b2,b2,b2,b1,b2,b2,b2,b1,
                       c1,c2,c2,c2,c1,c2,c2,c2,c1,
                       d1,d2,d3,d1,d3,d2,d2,d1,d3,d2,d3,d1,d3,d1,d2,d3,d2,d1])
        W = np.array([wa,
                        wb,wb,wb,
                        wc,wc,wc,
                        wd,wd,wd,wd,wd,wd])

    # 17 Gauss points
    if K==17:
        a = 1./3.
        b1 = 0.081414823414554; b2 = 0.459292588292723
        c1 = 0.658861384496480; c2 = 0.170569307751760
        d1 = 0.898905543365938; d2 = 0.050547228317031
        e1 = 0.008394777409958; e2 = 0.263112829634638; e3 = 0.728492392955404
        wa = 0.144315607677787
        wb = 0.095091634267285
        wc = 0.103217370534718
        wd = 0.032458497623198
        we = 0.027230314174435

        X = np.array([a,a,a,
                        b1,b2,b2,b2,b1,b2,b2,b2,b1,
                        c1,c2,c2,c2,c1,c2,c2,c2,c1,
                        d1,d2,d2,d2,d1,d2,d2,d2,d1,
                        e1,e2,e3,e1,e3,e2,e2,e1,e3,e2,e3,e1,e3,e1,e2,e3,e2,e1])
        W = np.array([wa,
                        wb,wb,wb,
                        wc,wc,wc,
                        wd,wd,wd,
                        we,we,we,we,we,we])

    # 19 Gauss points
    if K==19:
        a = 1./3.
        b1 = 0.020634961602525; b2 = 0.489682519198738
        c1 = 0.125820817014127; c2 = 0.437089591492937
        d1 = 0.623592928761935; d2 = 0.188203535619033
        e1 = 0.910540973211095; e2 = 0.044729513394453
        f1 = 0.036838412054736; f2 = 0.221962989160766; f3 = 0.741198598784498

        wa = 0.097135796282799
        wb = 0.031334700227139
        wc = 0.077827541004774
        wd = 0.079647738927210
        we = 0.025577675658698
        wf = 0.043283539377289

        X = np.array([a,a,a,
                        b1,b2,b2,b2,b1,b2,b2,b2,b1,
                        c1,c2,c2,c2,c1,c2,c2,c2,c1,
                        d1,d2,d2,d2,d1,d2,d2,d2,d1,
                        e1,e2,e2,e2,e1,e2,e2,e2,e1,
                        f1,f2,f3,f1,f3,f2,f2,f1,f3,f2,f3,f1,f3,f1,f2,f3,f2,f1])
        W = np.array([wa,
                        wb,wb,wb,
                        wc,wc,wc,
                        wd,wd,wd,
                        we,we,we,
                        wf,wf,wf,wf,wf,wf])

    # 25 Gauss points
    if K==25:
        a  = 1./3.
        b1 = 0.028844733232685; b2 = 0.485577633383657
        c1 = 0.781036849029926; c2 = 0.109481575485037
        d1 = 0.141707219414880; d2 = 0.307939838764121; d3 = 0.550352941820999
        e1 = 0.025003534762686; e2 = 0.246672560639903; e3 = 0.728323904597411
        f1 = 0.009540815400299; f2 = 0.066803251012200; f3 = 0.923655933587500

        wa = 0.090817990382754
        wb = 0.036725957756467
        wc = 0.045321059435528
        wd = 0.072757916845420
        we = 0.028327242531057
        wf = 0.009421666963733

        X = np.array([a,a,a,
                        b1,b2,b2,b2,b1,b2,b2,b2,b1,
                        c1,c2,c2,c2,c1,c2,c2,c2,c1,
                        d1,d2,d3,d1,d3,d2,d2,d1,d3,d2,d3,d1,d3,d1,d2,d3,d2,d1,
                        e1,e2,e3,e1,e3,e2,e2,e1,e3,e2,e3,e1,e3,e1,e2,e3,e2,e1,
                        f1,f2,f3,f1,f3,f2,f2,f1,f3,f2,f3,f1,f3,f1,f2,f3,f2,f1])
        W = np.array([wa,
                        wb,wb,wb,
                        wc,wc,wc,
                        wd,wd,wd,wd,wd,wd,
                        we,we,we,we,we,we,
                        wf,wf,wf,wf,wf,wf])

    # 37 Gauss points
    if K==37:
        a = 1./3.
        b1 = 0.009903630120591; b2 = 0.495048184939705
        c1 = 0.062566729780852; c2 = 0.468716635109574
        d1 = 0.170957326397447; d2 = 0.414521336801277
        e1 = 0.541200855914337; e2 = 0.229399572042831
        f1 = 0.771151009607340; f2 = 0.114424495196330
        g1 = 0.950377217273082; g2 = 0.024811391363459
        h1 = 0.094853828379579; h2 = 0.268794997058761; h3 = 0.636351174561660
        i1 = 0.018100773278807; i2 = 0.291730066734288; i3 = 0.690169159986905
        j1 = 0.022233076674090; j2 = 0.126357385491669; j3 = 0.851409537834241

        wa = 0.052520923400802
        wb = 0.011280145209330
        wc = 0.031423518362454
        wd = 0.047072502504194
        we = 0.047363586536355
        wf = 0.031167529045794
        wg = 0.007975771465074
        wh = 0.036848402728732
        wi = 0.017401463303822
        wj = 0.015521786839045

        X = np.array([a,a,a,
                         b1,b2,b2,b2,b1,b2,b2,b2,b1,
                         c1,c2,c2,c2,c1,c2,c2,c2,c1,
                         d1,d2,d2,d2,d1,d2,d2,d2,d1,
                         e1,e2,e2,e2,e1,e2,e2,e2,e1,
                         f1,f2,f2,f2,f1,f2,f2,f2,f1,
                         g1,g2,g2,g2,g1,g2,g2,g2,g1,
                         h1,h2,h3,h1,h3,h2,h2,h1,h3,h2,h3,h1,h3,h1,h2,h3,h2,h1,
                         i1,i2,i3,i1,i3,i2,i2,i1,i3,i2,i3,i1,i3,i1,i2,i3,i2,i1,
                         j1,j2,j3,j1,j3,j2,j2,j1,j3,j2,j3,j1,j3,j1,j2,j3,j2,j1])
        W = np.array([wa,
                         wb,wb,wb,
                         wc,wc,wc,
                         wd,wd,wd,
                         we,we,we,
                         wf,wf,wf,
                         wg,wg,wg,
                         wh,wh,wh,wh,wh,wh,
                         wi,wi,wi,wi,wi,wi,
                         wj,wj,wj,wj,wj,wj])

    # 48 Gauss points
    if K==48:
        a1 =-0.013945833716486; a2 = 0.506972916858243
        b1 = 0.137187291433955; b2 = 0.431406354283023
        c1 = 0.444612710305711; c2 = 0.277693644847144
        d1 = 0.747070217917492; d2 = 0.126464891041254
        e1 = 0.858383228050628; e2 = 0.070808385974686
        f1 = 0.962069659517853; f2 = 0.018965170241073
        g1 = 0.133734161966621; g2 = 0.261311371140087; g3 = 0.604954466893291
        h1 = 0.036366677396917; h2 = 0.388046767090269; h3 = 0.575586555512814
        i1 =-0.010174883126571; i2 = 0.285712220049916; i3 = 0.724462663076655
        j1 = 0.036843869875878; j2 = 0.215599664072284; j3 = 0.747556466051838
        k1 = 0.012459809331199; k2 = 0.103575616576386; k3 = 0.883964574092416

        wa = 0.001916875642849
        wb = 0.044249027271145
        wc = 0.051186548718852
        wd = 0.023687735870688
        we = 0.013289775690021
        wf = 0.004748916608192
        wg = 0.038550072599593
        wh = 0.027215814320624
        wi = 0.002182077366797
        wj = 0.021505319847731
        wk = 0.007673942631049

        X = np.array([a1,a2,a2,a2,a1,a2,a2,a2,a1,
                         b1,b2,b2,b2,b1,b2,b2,b2,b1,
                         c1,c2,c2,c2,c1,c2,c2,c2,c1,
                         d1,d2,d2,d2,d1,d2,d2,d2,d1,
                         e1,e2,e2,e2,e1,e2,e2,e2,e1,
                         f1,f2,f2,f2,f1,f2,f2,f2,f1,
                         g1,g2,g3,g1,g3,g2,g2,g1,g3,g2,g3,g1,g3,g1,g2,g3,g2,g1,
                         h1,h2,h3,h1,h3,h2,h2,h1,h3,h2,h3,h1,h3,h1,h2,h3,h2,h1,
                         i1,i2,i3,i1,i3,i2,i2,i1,i3,i2,i3,i1,i3,i1,i2,i3,i2,i1,
                         j1,j2,j3,j1,j3,j2,j2,j1,j3,j2,j3,j1,j3,j1,j2,j3,j2,j1,
                         k1,k2,k3,k1,k3,k2,k2,k1,k3,k2,k3,k1,k3,k1,k2,k3,k2,k1])
        W = np.array([wa,wa,wa,
                         wb,wb,wb,
                         wc,wc,wc,
                         wd,wd,wd,
                         we,we,we,
                         wf,wf,wf,
                         wg,wg,wg,wg,wg,wg,
                         wh,wh,wh,wh,wh,wh,
                         wi,wi,wi,wi,wi,wi,
                         wj,wj,wj,wj,wj,wj,
                         wk,wk,wk,wk,wk,wk])

    # 52 Gauss points
    if K==52:
        a = 1./3.
        b1 = 0.005238916103123; b2 = 0.497380541948438
        c1 = 0.173061122901295; c2 = 0.413469438549352
        d1 = 0.059082801866017; d2 = 0.470458599066991
        e1 = 0.518892500060958; e2 = 0.240553749969521
        f1 = 0.704068411554854; f2 = 0.147965794222573
        g1 = 0.849069624685052; g2 = 0.075465187657474
        h1 = 0.966807194753950; h2 = 0.016596402623025
        i1 = 0.103575692245252; i2 = 0.296555596579887; i3 = 0.599868711174861
        j1 = 0.020083411655416; j2 = 0.337723063403079; j3 = 0.642193524941505
        k1 =-0.004341002614139; k2 = 0.204748281642812; k3 = 0.799592720971327
        l1 = 0.041941786468010; l2 = 0.189358492130623; l3 = 0.768699721401368
        m1 = 0.014317320230681; m2 = 0.085283615682657; m3 = 0.900399064086661

        wa = 0.046875697427642
        wb = 0.006405878578585
        wc = 0.041710296739387
        wd = 0.026891484250064
        we = 0.042132522761650
        wf = 0.030000266842773
        wg = 0.014200098925024
        wh = 0.003582462351273
        wi = 0.032773147460627
        wj = 0.015298306248441
        wk = 0.002386244192839
        wl = 0.019084792755899
        wm = 0.006850054546542

        X = np.array([a,a,a,
                         b1,b2,b2,b2,b1,b2,b2,b2,b1,
                         c1,c2,c2,c2,c1,c2,c2,c2,c1,
                         d1,d2,d2,d2,d1,d2,d2,d2,d1,
                         e1,e2,e2,e2,e1,e2,e2,e2,e1,
                         f1,f2,f2,f2,f1,f2,f2,f2,f1,
                         g1,g2,g2,g2,g1,g2,g2,g2,g1,
                         h1,h2,h2,h2,h1,h2,h2,h2,h1,
                         i1,i2,i3,i1,i3,i2,i2,i1,i3,i2,i3,i1,i3,i1,i2,i3,i2,i1,
                         j1,j2,j3,j1,j3,j2,j2,j1,j3,j2,j3,j1,j3,j1,j2,j3,j2,j1,
                         k1,k2,k3,k1,k3,k2,k2,k1,k3,k2,k3,k1,k3,k1,k2,k3,k2,k1,
                         l1,l2,l3,l1,l3,l2,l2,l1,l3,l2,l3,l1,l3,l1,l2,l3,l2,l1,
                         m1,m2,m3,m1,m3,m2,m2,m1,m3,m2,m3,m1,m3,m1,m2,m3,m2,m1])
        W = np.array([wa,
                         wb,wb,wb,
                         wc,wc,wc,
                         wd,wd,wd,
                         we,we,we,
                         wf,wf,wf,
                         wg,wg,wg,
                         wh,wh,wh,
                         wi,wi,wi,wi,wi,wi,
                         wj,wj,wj,wj,wj,wj,
                         wk,wk,wk,wk,wk,wk,
                         wl,wl,wl,wl,wl,wl,
                         wm,wm,wm,wm,wm,wm])

    # 61 Gauss points
    if K==61:
        a = 1./3.
        b1 = 0.005658918886452; b2 = 0.497170540556774
        c1 = 0.035647354750751; c2 = 0.482176322624625
        d1 = 0.099520061958437; d2 = 0.450239969020782
        e1 = 0.199467521245206; e2 = 0.400266239377397
        f1 = 0.495717464058095; f2 = 0.252141267970953
        g1 = 0.675905990683077; g2 = 0.162047004658461
        h1 = 0.848248235478508; h2 = 0.075875882260746
        i1 = 0.968690546064356; i2 = 0.015654726967822
        j1 = 0.010186928826919; j2 = 0.334319867363658; j3 = 0.655493203809423
        k1 = 0.135440871671036; k2 = 0.292221537796944; k3 = 0.572337590532020
        l1 = 0.054423924290583; l2 = 0.319574885423190; l3 = 0.626001190286228
        m1 = 0.012868560833637; m2 = 0.190704224192292; m3 = 0.796427214974071
        n1 = 0.067165782413524; n2 = 0.180483211648746; n3 = 0.752351005937729
        o1 = 0.014663182224828; o2 = 0.080711313679564; o3 = 0.904625504095608

        wa = 0.033437199290803
        wb = 0.005093415440507
        wc = 0.014670864527638
        wd = 0.024350878353672
        we = 0.031107550868969
        wf = 0.031257111218620
        wg = 0.024815654339665
        wh = 0.014056073070557
        wi = 0.003194676173779
        wj = 0.008119655318993
        wk = 0.026805742283163
        wl = 0.018459993210822
        wm = 0.008476868534328
        wn = 0.018292796770025
        wo = 0.006665632004165

        X = np.array([a,a,a,
                         b1,b2,b2,b2,b1,b2,b2,b2,b1,
                         c1,c2,c2,c2,c1,c2,c2,c2,c1,
                         d1,d2,d2,d2,d1,d2,d2,d2,d1,
                         e1,e2,e2,e2,e1,e2,e2,e2,e1,
                         f1,f2,f2,f2,f1,f2,f2,f2,f1,
                         g1,g2,g2,g2,g1,g2,g2,g2,g1,
                         h1,h2,h2,h2,h1,h2,h2,h2,h1,
                         i1,i2,i2,i2,i1,i2,i2,i2,i1,
                         j1,j2,j3,j1,j3,j2,j2,j1,j3,j2,j3,j1,j3,j1,j2,j3,j2,j1,
                         k1,k2,k3,k1,k3,k2,k2,k1,k3,k2,k3,k1,k3,k1,k2,k3,k2,k1,
                         l1,l2,l3,l1,l3,l2,l2,l1,l3,l2,l3,l1,l3,l1,l2,l3,l2,l1,
                         m1,m2,m3,m1,m3,m2,m2,m1,m3,m2,m3,m1,m3,m1,m2,m3,m2,m1,
                         n1,n2,n3,n1,n3,n2,n2,n1,n3,n2,n3,n1,n3,n1,n2,n3,n2,n1,
                         o1,o2,o3,o1,o3,o2,o2,o1,o3,o2,o3,o1,o3,o1,o2,o3,o2,o1])
        W = np.array([wa,
                         wb,wb,wb,
                         wc,wc,wc,
                         wd,wd,wd,
                         we,we,we,
                         wf,wf,wf,
                         wg,wg,wg,
                         wh,wh,wh,
                         wi,wi,wi,
                         wj,wj,wj,wj,wj,wj,
                         wk,wk,wk,wk,wk,wk,
                         wl,wl,wl,wl,wl,wl,
                         wm,wm,wm,wm,wm,wm,
                         wn,wn,wn,wn,wn,wn,
                         wo,wo,wo,wo,wo,wo])

    # 79 Gauss points
    if K==79:
        a  = 1./3.
        b1 = -0.001900928704400; b2 = 0.500950464352200
        c1 = 0.023574084130543; c2 = 0.488212957934729
        d1 = 0.089726636099435; d2 = 0.455136681950283
        e1 = 0.196007481363421; e2 = 0.401996259318289
        f1 = 0.488214180481157; f2 = 0.255892909759421
        g1 = 0.647023488009788; g2 = 0.176488255995106
        h1 = 0.791658289326483; h2 = 0.104170855336758
        i1 = 0.893862072318140; i2 = 0.053068963840930
        j1 = 0.916762569607942; j2 = 0.041618715196029
        k1 = 0.976836157186356; k2 = 0.011581921406822
        l1 = 0.048741583664839; l2 = 0.344855770229001; l3 = 0.606402646106160
        m1 = 0.006314115948605; m2 = 0.377843269594854; m3 = 0.615842614456541
        n1 = 0.134316520547348; n2 = 0.306635479062357; n3 = 0.559048000390295
        o1 = 0.013973893962392; o2 = 0.249419362774742; o3 = 0.736606743262866
        p1 = 0.075549132909764; p2 = 0.212775724802802; p3 = 0.711675142287434
        q1 = -0.008368153208227; q2 = 0.146965436053239; q3 = 0.861402717154987
        r1 = 0.026686063258714; r2 = 0.137726978828923; r3 = 0.835586957912363
        s1 = 0.010547719294141; s2 = 0.059696109149007; s3 = 0.929756171556853

        wa = 0.033057055541624
        wb = 0.000867019185663
        wc = 0.011660052716448
        wd = 0.022876936356421
        we = 0.030448982673938
        wf = 0.030624891725355
        wg = 0.024368057676800
        wh = 0.015997432032024
        wi = 0.007698301815602
        wj = -0.000632060497488
        wk = 0.001751134301193
        wl = 0.016465839189576
        wm = 0.004839033540485
        wn = 0.025804906534650
        wo = 0.008471091054441
        wp = 0.018354914106280
        wq = 0.000704404677908
        wr = 0.010112684927462
        ws = 0.003573909385950

        X = np.array([a,a,a,
                         b1,b2,b2,b2,b1,b2,b2,b2,b1,
                         c1,c2,c2,c2,c1,c2,c2,c2,c1,
                         d1,d2,d2,d2,d1,d2,d2,d2,d1,
                         e1,e2,e2,e2,e1,e2,e2,e2,e1,
                         f1,f2,f2,f2,f1,f2,f2,f2,f1,
                         g1,g2,g2,g2,g1,g2,g2,g2,g1,
                         h1,h2,h2,h2,h1,h2,h2,h2,h1,
                         i1,i2,i2,i2,i1,i2,i2,i2,i1,
                         j1,j2,j2,j2,j1,j2,j2,j2,j1,
                         k1,k2,k2,k2,k1,k2,k2,k2,k1,
                         l1,l2,l3,l1,l3,l2,l2,l1,l3,l2,l3,l1,l3,l1,l2,l3,l2,l1,
                         m1,m2,m3,m1,m3,m2,m2,m1,m3,m2,m3,m1,m3,m1,m2,m3,m2,m1,
                         n1,n2,n3,n1,n3,n2,n2,n1,n3,n2,n3,n1,n3,n1,n2,n3,n2,n1,
                         o1,o2,o3,o1,o3,o2,o2,o1,o3,o2,o3,o1,o3,o1,o2,o3,o2,o1,
                         p1,p2,p3,p1,p3,p2,p2,p1,p3,p2,p3,p1,p3,p1,p2,p3,p2,p1,
                         q1,q2,q3,q1,q3,q2,q2,q1,q3,q2,q3,q1,q3,q1,q2,q3,q2,q1,
                         r1,r2,r3,r1,r3,r2,r2,r1,r3,r2,r3,r1,r3,r1,r2,r3,r2,r1,
                         s1,s2,s3,s1,s3,s2,s2,s1,s3,s2,s3,s1,s3,s1,s2,s3,s2,s1])

        W = np.array([wa,
                         wb,wb,wb,
                         wc,wc,wc,
                         wd,wd,wd,
                         we,we,we,
                         wf,wf,wf,
                         wg,wg,wg,
                         wh,wh,wh,
                         wi,wi,wi,
                         wj,wj,wj,
                         wk,wk,wk,
                         wl,wl,wl,wl,wl,wl,
                         wm,wm,wm,wm,wm,wm,
                         wn,wn,wn,wn,wn,wn,
                         wo,wo,wo,wo,wo,wo,
                         wp,wp,wp,wp,wp,wp,
                         wq,wq,wq,wq,wq,wq,
                         wr,wr,wr,wr,wr,wr,
                         ws,ws,ws,ws,ws,ws])
    return(X, W)

def evaluation_points_and_weights_new(v1,v2,v3 , N , X_K , W):
    
    x_k_fun = lambda i : np.dot(X_K[i] , (v1,v2,v3) )
    x_k =  np.array([ x_k_fun(i) for i in range(N)]) 
    return x_k

def saved_sphere_distributions(name , r):
    '''
    Useful when running spherical grids.
    Inputs
    name  : Name of the distribution. Can be 'sphere_cent', 'sphere_offcent' or 'charge-dipole'.
    r     : Sphere radius
    Returns x_q , q 
    '''
        
    if name == 'sphere_cent':
        x_q = np.array( [[  1.E-12 ,  1.E-12 ,  1.E-12 ]]  )
        q = np.array( [1.] )
    
    if name == 'sphere_offcent':
        x_q = np.array( [[  1.E-12 ,  1.E-12 ,   r/2. ]]  )
        q = np.array( [1.] )
        
    if name == 'charge-dipole':
        x_q = np.array( [[  1.E-12 ,  1.E-12 ,  0.62 ],
                 [  1.E-12 ,  0.62*np.cos(np.pi*1.5 + 5.*np.pi/180.) ,
                                                      0.62*np.sin(np.pi*1.5 + 5.*np.pi/180. ) ] ,
                 [  1.E-12 ,  0.62*np.cos(np.pi*1.5 - 5.*np.pi/180.) ,
                                                      0.62*np.sin(np.pi*1.5 - 5.*np.pi/180. )  ]
                       ] )
        q = np.array( [1. , 1. , -1.]) 
    
    return x_q , q

def integral_per_element(f , df , face_array , vert_array , f_space , f_order , df_space , df_order ,
                         N , grid = None):
    '''
    Returns 2 arrays containing the integral per element of f and df.
    f : Solution from Bempp
    df: Solution from Bempp
    face_array starting from 0
    vert_array
    f_space
    f_order
    df_space
    df_order
    grid : The grid
    '''
    if grid == None:
        print('Fatal error, didnt find the grid.')
        
    f_array = np.empty((0,1))
    df_array= np.empty((0,1))
    
    sol_ones = np.ones((N+1 , 1))
    
    const_space = bempp.api.function_space(grid, "DP", 0)
    ones = bempp.api.GridFunction(const_space, coefficients=np.ones((len(face_array),)))
    
    for face in face_array:
        f_aux = int_calc_i( face+1  , face_array+1 , vert_array ,
                           f ,  f_space ,  f_order , ones , 'DP' , 0 , N)
        
        df_aux= int_calc_i( face+1  , face_array+1  , vert_array ,
                           df , df_space , df_order , ones , 'DP' , 0 , N)
        
        f_array  = np.vstack((f_array  , f_aux ))
        df_array = np.vstack((df_array , df_aux))
        
    return f_array , df_array

def matrix_lineal_transform( v1 , v2 , v3 ):
    '''
    Returns the asociated matrix transf. for the linear system
                                            (0,1) eta
                         | (0 , 0 , 0 ) |           |\
      [A] | v1,v2,v3 | = | (1 , 0 , 0 ) |           | \   <---- Transformed triangle
                         | (0 , 1 , 0 ) |     (0,0) |__\____ Xi
                                                       (1,0) 
    '''
    
    V =   np.transpose( np.array( ( v1 , v2 , v3) ) )
    
    if np.linalg.det(V) == 0:
        print(V)
        print('No invertible matrix encountered!')
        
    
    TL =  np.array( (
        [ 0. , 0. , 0. ] , 
        [ 1. , 0. , 0. ] , 
        [ 0. , 1. , 0. ]
             ) )
    
    A = np.dot( np.transpose(TL) , np.linalg.inv(V) )
    
    return A

def local_f( x , A , s1s2s3 , order):
    '''
    Estimates value for a given solution in vertices.
    x     : evaluated point
    A     : Asociated LT matrix
    s1s2s3: Sorted solution for v1,v2 and v3
    order : Order of the space solution
    '''
    if order == 0:
        value_in_x = s1s2s3
        
    elif order == 1:
        xi , eta = linear_weights( x, A)

        N = N_values(xi,eta,order)

        value_in_x = np.sum( N * s1s2s3)
    else:
        print('Plugin not ready!')
        value_in_x = 0
    return value_in_x

def linear_weights( x , A):
    eta , Xi , _ = np.dot( A , x )
    #Xi , eta , _ = np.dot( A , x )
    return Xi , eta

def N_values( Xi , eta , order):
    '''
    Returns coefficients to interpolate linear basis functions
    
    '''
    if order == 1:
        N1 = 1. - Xi - eta
        N2 = Xi
        N3 = eta
        
        #print(N1,N2,N3)

        return N1 , N2 , N3

    elif order == 2:
        N1 = (1. - Xi - eta) * (1. - 2.* Xi - 2.* eta)
        N2 = Xi  * (2.* Xi - 1 )
        N3 = eta * (2. * eta -1)
        N4 = 4. * Xi * (1. - Xi - eta)
        N5 = 4. * Xi * eta
        N6 = 4. * eta * (1. - Xi - eta)
        
        return N1,N2,N3,N4,N5,N6
    
    else:
        print('Error en funcion N_values')
        return None
