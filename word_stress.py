#!/usr/bin/env python3
# coding: utf-8

from __future__ import division, print_function
import re

# Vowel Phonemes
import nltk

vowels = (
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
    'IH', 'IY', 'OW', 'OY', 'UH', 'UW')

vowels_set = set(vowels)


# Consonants Phonemes
consonants = (
    'P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M',
    'N', 'NG', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH')

consonants_set = set(consonants)


std_phonemes = vowels + consonants
std_phonemees_map = {phoneme.upper(): ix for ix, phoneme in enumerate(std_phonemes)}

suffixes = """
    inal tion sion osis oon sce que ette eer ee aire able ible
    acy cy ade age al al ial ical an ance ence ancy ency ant ent
    ant ent ient ar ary ard art ate ate ate ation cade drome ed
    ed en en ence ency er ier er or er or ery es ese ies es ies
    ess est iest fold ful ful fy ia ian iatry ic ic ice ify ile
    ing ion ish ism ist ite ity ive ive ative itive ize less ly
    ment ness or ory ous eous ose ious ship ster ure ward wise
    ize phy ogy""".split()


prefixes = """
    ac ad af ag al an ap as at an ab abs acer acid acri act ag
    acu aer aero ag agi ig act agri agro alb albo ali allo alter
    alt am ami amor ambi ambul ana ano andr andro ang anim ann
    annu enni ante anthrop anti ant anti antico apo ap aph aqu
    arch aster astr auc aug aut aud audi aur aus aug auc aut
    auto bar be belli bene bi bine bibl bibli biblio bio bi brev
    cad cap cas ceiv cept capt cid cip cad cas calor capit capt
    carn cat cata cath caus caut cause cuse cus ceas ced cede
    ceed cess cent centr centri chrom chron cide cis cise circum
    cit civ clam claim clin clud clus claus co cog col coll con
    com cor cogn gnos com con contr contra counter cord cor
    cardi corp cort cosm cour cur curr curs crat cracy cre cresc
    cret crease crea cred cresc cret crease cru crit cur curs
    cura cycl cyclo de dec deca dec dign dei div dem demo dent
    dont derm di dy dia dic dict dit dis dif dit doc doct domin
    don dorm dox duc duct dura dynam dys ec eco ecto en em end
    epi equi erg ev et ex exter extra extro fa fess fac fact fec
    fect fic fas fea fall fals femto fer fic feign fain fit feat
    fid fid fide feder fig fila fili fin fix flex flect flict
    flu fluc fluv flux for fore forc fort form fract frag frai
    fuge fuse gam gastr gastro gen gen geo germ gest giga gin
    gloss glot glu glo gor grad gress gree graph gram graf grat
    grav greg hale heal helio hema hemo her here hes hetero hex
    ses sex homo hum human hydr hydra hydro hyper hypn an ics
    ignis in im in im il ir infra inter intra intro ty jac ject
    join junct judice jug junct just juven labor lau lav lot lut
    lect leg lig leg levi lex leag leg liber liver lide liter
    loc loco log logo ology loqu locut luc lum lun lus lust lude
    macr macer magn main mal man manu mand mania mar mari mer
    matri medi mega mem ment meso meta meter metr micro migra
    mill kilo milli min mis mit miss mob mov mot mon mono mor
    mort morph multi nano nasc nat gnant nai nat nasc neo neur
    nom nom nym nomen nomin non non nov nox noc numer numisma ob
    oc of op oct oligo omni onym oper ortho over pac pair pare
    paleo pan para pat pass path pater patr path pathy ped pod
    pedo pel puls pend pens pond per peri phage phan phas phen
    fan phant fant phe phil phlegma phobia phobos phon phot
    photo pico pict plac plais pli ply plore plu plur plus
    pneuma pneumon pod poli poly pon pos pound pop port portion
    post pot pre pur prehendere prin prim prime pro proto psych
    punct pute quat quad quint penta quip quir quis quest quer
    re reg recti retro ri ridi risi rog roga rupt sacr sanc secr
    salv salu sanct sat satis sci scio scientia scope scrib
    script se sect sec sed sess sid semi sen scen sent sens sept
    sequ secu sue serv sign signi simil simul sist sta stit soci
    sol solus solv solu solut somn soph spec spect spi spic sper
    sphere spir stand stant stab stat stan sti sta st stead
    strain strict string stige stru struct stroy stry sub suc
    suf sup sur sus sume sump super supra syn sym tact tang tag
    tig ting tain ten tent tin tect teg tele tem tempo ten tin
    tain tend tent tens tera term terr terra test the theo therm
    thesis thet tire tom tor tors tort tox tract tra trai treat
    trans tri trib tribute turbo typ ultima umber umbraticum un
    uni vac vade vale vali valu veh vect ven vent ver veri verb
    verv vert vers vi vic vicis vict vinc vid vis viv vita vivi
    voc voke vol volcan volv volt vol vor with zo""".split()


suffixes_map = {suffix.upper(): ix for ix, suffix in enumerate(suffixes)}
prefixes_map = {prefix.upper(): ix for ix, prefix in enumerate(prefixes)}


def parse_phoneme(s):
    """
    >>> parse_phoneme('AH0') 
    ('AH', 0)
    >>> parse_phoneme('K') 
    ('K', -1)
    :param s: 
    :return: 
    """
    mat = re.match(r'([A-Za-z]+)(\d)', s)
    if not mat:
        return
    ph, stress = mat.groups()
    return ph, int(stress)


def parse_line(line):
    """
    >>> parse('NONPOISONOUS:N AA0 N P OY1 Z AH0 N AH0 S')
    """
    parts = re.split(r'[\s:]+', line)
    if not parts:
        return
    word = parts[0]
    pairs = [parse_phoneme(p) for p in parts[1:]]
    phonemes = [x[0] for x in pairs]
    stresses = [x[1] for x in pairs]
    return word, phonemes, stresses


def parse_training_data(path):
    with open(path) as fin:
        for line in fin:
            yield parse_line(line)


def get_phoneme_number(phoneme):
    return std_phonemees_map.get(phoneme, -1)


def get_prefix_number(word):
    """
    >>> get_prefix_number('advertize')
    1
    :param word: 
    :return: 
    """
    for i in range(len(word)):
        x = word[:i + 1]
        if x in prefixes_map:
            return prefixes_map[x]
    return -1


def get_suffix_number(word):
    """
    >>> get_suffix_number('final') 
    0
    :param word: 
    :return: 
    """
    for i in range(len(word)):
        x = word[-1 - i:]
        if x in suffixes_map:
            return suffixes_map[x]
    return -1


# Get nltk pos_tag
def get_pos_tag(word):
    return nltk.pos_tag([word])[0][1]


def get_features(word, phonemes, phoneme_ix):
    vowels_count = len([x for x in phonemes if x in vowels_set])
    consonants_count = len(phonemes) - vowels_count
    return {
        'phoneme_number': get_phoneme_number(phonemes[phoneme_ix][0]),
        'prefix_number': get_prefix_number(word),
        'suffix_number': get_suffix_number(word),
        'pos_tag': get_pos_tag(word),
        'phonemes_count': len(phonemes),
        'vowels_count': vowels_count,
        'consonants_count': consonants_count,
    }

