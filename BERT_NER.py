from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# Initialize the pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Full legal text input
example = """


SECOND DIVISION
[ G.R. No. 258887, July 31, 2023 ]
LUZ DELOS SANTOS, MINORS FRANCIS DELOS SANTOS, CATHERINE DELOS SANTOS, AND LORENCE DELOS SANTOS, PETITIONERS, VS. DEMY ALMA M. DELOS SANTOS, MONTANO M. DELOS SANTOS JOINED BY HIS WIFE ANALIZA G. DELOS SANTOS, IRENE ANGELA D. CLEMENTE JOINED BY HER HUSBAND ANGELO CLEMENTE, AND SEATIEL M. DELOS SANTOS JOINED BY HIS WIFE MICHELLE R. DELOS SANTOS, RESPONDENTS.

D E C I S I O N
LOPEZ, M., J.:

Quando res non valet ut ago, valeat quantum valere potest: "a contract must be recognized as far as it is legally possible to do so."[1] Thus, while an extrajudicial settlement or conveyance which excluded co-heirs of their rightful share in the inheritance is void and inexistent,[2] transfers pertaining to the undivided share of the conveying co-heir should be recognized as valid subject to proper liquidation and partition.[3]

This resolves the Petition for Review on Certiorari[4] under Rule 45 of the Rules of Court filed by Luz Delos Santos (Luz) with her children, namely: Francis Delos Santos (Francis), Catherine Delos Santos (Catherine), and Lorence Delos Santos (Lorence), all surnamed Delos Santos, assailing the Decision[5] dated December 4, 2020 and Resolution[6] dated November 26, 2021 of the Court of Appeals (CA) in CA-G.R. CV No. 111116. The assailed CA issuances affirmed the Decision[7] dated October 5, 2017 of the Regional Trial Court (RTC) of Olongapo City, Branch 72, in Civil Case No. 26-0-2012.

Facts

Subject of this case are the conjugal partnership properties of Spouses Emerenciano Delos Santos (Emerenciano) and Adalia Delos Santos (Adalia), consisting of two parcels of land with improvements in Olongapo City, covered by: (1) Transfer Certificate of Title (TCT) No. T-3337,[8] declared for tax purposes under Tax Declaration (TD) No. AB00302198[9] (land) and TD No. AB00302199[10] (improvement); and (2) Original Certificate of Title (OCT) No. P-1850,[11] declared for tax purposes under TD No. AB003016666[12] (land) and TD Nos. AB00301667[13] and AB00301666[14] (improvements).

Adalia died in 1996, survived by Emerenciano and their natural child, Demy Alma Delos Santos (Demy) and adopted children, Montano Delos Santos (Montano), Irene Angela D. Clemente (Irene), and Seatiel Delos Santos (Seatiel). About seven years after Adalia's passing, Emerenciano married Luz with whom he begot three children: Francis, Catherine, and Lorence.[15]

In 2009, Emerenciano and Luz, representing then minors Francis, Catherine, and Lorence, executed an Extrajudicial Settlement of Estate with Waiver[16] (EJSW) concerning the subject properties. In the EJSW, Emerenciano and his minor children represented that they are the sole heirs of Adalia. Emerenciano then adjudicated to himself one-half portion of the properties as his conjugal share, and thereafter, conveyed the entirety of the properties as follows: (1) the land covered by OCT No. P-1850, including the residential house on it, equally in favor of Catherine and Lorence; and (2) the land covered by TCT No. T-3337, including the residential house on it, equally in favor of Francis, Catherine, and Lorence. The following year, Emerenciano also executed a Deed of Waiver, Quitclaim and Transfer of Residential Buildings, conveying the residential buildings covered by TD Nos. AB00301667 and AB00301668 in favor of Catherine and Lorence.[17] Consequently, TCT No. 15439, in lieu of TCT No. T-3337, and TCT No. T-15438, in lieu of OCT No. P-1850, were issued in the names of Francis, Catherine, and Lorence. The original TDs were also cancelled, and new ones were issued accordingly.[18]

In 2011, Emerenciano died. Demy, Montano, Irene, and Seatiel discovered the conveyances made by their father in favor of Francis, Catherine, and Lorence. Unsuccessful in settling the controversy among themselves, Demy, Montano, Irene, and Seatiel filed a Complaint[19] for the annulment of the EJSW and the Deed of Waiver, Quitclaim and Transfer of Residential Buildings, and the cancellation of the new TCTs and TDs issued by virtue of said deeds. They claimed that the conveyed properties were conjugal properties of their parents, Emerenciano and Adalia. Hence, the conveyances made to their exclusion were void as they were deprived of their rightful shares as legal heirs of Adalia. They further pointed out that Emerenciano and his then minor children, represented by their mother Luz, grossly misrepresented themselves in the EJSW as the only heirs of Adalia. They pointed out that Francis, Catherine, and Lorence are not related in any way to Adalia, and hence, are not entitled to Adalia's share in the conjugal properties. Finally, they argued that their mother's estate should have been settled first before their father alienated his share from the conjugal properties in favor of his children from his second marriage.

Luz and her children, on the other hand, attacked the filiation of Demy, Montano, Irene, and Seatiel with Emerenciano and Adalia. They alleged that Demy is not Emerenciano and Adalia's natural child, but merely adopted like Montano, Irene, and Seatiel. They further pointed out that Demy, Montano, Irene, and Seatiel were fully aware of the questioned conveyances as they were also given other properties. Hence, they supposedly have no basis to complain since all Emerenciano's children already received their respective inheritance from their father.

In its Decision,[20] the RTC emphasized that the properties in controversy are conjugal properties of Emerenciano and Adalia. The RTC then recognized Demy, Montano, Irene, and Seatiel as legal heirs entitled to their legitimes from Adalia's estate, being natural and adopted children of Emerenciano and Adalia. Therefore, the deeds of conveyances executed by Emerenciano in favor of his children from his second marriage to the exclusion of Demy, Montano, Irene, and Seatiel are void. The RTC, however, clarified that the conveyances were not altogether invalid as Emerenciano's children from Luz may be entitled to the free portion of the conveyed properties pertaining to Emerenciano's share. But despite such clarification, the RTC annulled the EJSW and Deed of Waiver, Quitclaim and Transfer of Residential Buildings altogether, thus:
a) Recognizing and appointing [LUZ] to be the legal guardian over her minor children [FRANCIS, CATHERINE, and LORENCE] in the case in order to protect their proprietary rights in the subject properties;

b) Annulling the [EJSW] x x x;

c) Annulling the Deed of Waiver, Quitclaim and Transfer of Residential Building[s] x x x;

d) Directing the Register of Deeds of Olongapo City to cancel [TCT] No. T-15438 x x x to revert it back [sic] to (OCT] No. P-1850 x x x;

e) Directing the Register of Deeds of Olongapo City to cancel [TCT] No. T-15439 pertaining only to the half portion belonging to the deceased and revert it back [sic] to [TCT] NO. T-3337 x x x; and

f) Directing the Office of the City Assessor of Olongapo to: a.) Cancel [TD] No. AB00303343 and revert it to [TD] No. AB00302198 x x x; b.) Cancel [TD] No. AB00303344 and revert it to [TD] No. AB00301666 x x x; c.) Cancel [TD] No. AB00303331 and revert it to [TD] No. AB00302199 x x x; d.) Cancel [TD] No. AB00303332 and revert it to [TD] No. AB00301667 x x x; f.) [sic] Cancel [TD] No. AB00303333 and revert it to [TD] No. AB00301668 x x x.

SO ORDERED.[21] (Emphasis in the original)
Luz and her children appealed to the CA with the same arguments, adding that laches and prescription have already set in since Demy, Montano, Irene, and Seatiel are questioning a deceased person's act. In its assailed Decision,[22] the CA denied the appeal and affirmed the RTC ruling in toto. The Motion for Reconsideration, filed by Luz and her children, was likewise denied in the assailed Resolution.[23] Hence, this Petition.

In this recourse, petitioners Luz, Francis, Catherine, and Lorence fault the RTC for concluding that Demy, Montano, Irene, and Seatiel are Adalia's legal heirs. Petitioners argue that filiation cannot be determined in an ordinary case for annulment of documents as such matter should first be determined in a separate special proceeding.[24] In any case, petitioners maintain that respondents were aware of Emerenciano's conveyances as they also received properties from their father. Hence, petitioners posit that respondents are estopped from questioning the extrajudicial settlement, as well as the conveyances.[25] Petitioners also contend that laches and prescription have already set in since respondents failed to lodge their objection during Emerenciano's lifetime.[26] Finally, petitioners claim that the EJSW should be interpreted to give effect to Emerenciano's intention to give his share in the conjugal properties to Francis, Catherine, and Lorence.[27]

Issues

As synthesized, the Court shall resolve the following issues, viz.:
Is a separate proceeding necessary before the trial court can acknowledge respondents' heirship?

Did the courts a quo err in nullifying the EJSW and the Deed of Waiver, Quitclaim and Transfer of Residential Buildings, as well as the TCTs and TDs issued by virtue of the nullified deeds?

Was respondents' cause of action already barred by laches and prescription?
Ruling

The Petition is partly meritorious.
 
A separate proceeding is not necessary to acknowledge respondents as Adalia's legal heirs
 

It must be emphasized at the outset that, in their Complaint, respondents assert their right as the natural and adopted children of their deceased mother and seek to annul deeds of extrajudicial settlement and conveyances which deprived them of their hereditary rights. They are not seeking to establish their heirship since they already possess such status and right by virtue of law.[28] Indeed, as Adalia's children, respondents are Adalia's legal heirs by operation of law.[29]

Hence, a separate proceeding is not necessary for the RTC to acknowledge them as Adalia's legal heirs.[30]

Besides, it was petitioners who raised questions on respondents' succession rights, claiming that Demy is not a natural child but merely adopted like Montano, Irene, and Seatiel. Hence, at this point, petitioners cannot validly question the RTC's jurisdiction to rule on the matter that they themselves raised. It is settled that a separate special proceeding for the determination of heirship may be dispensed with for the sake of practicality when the parties in the civil case had voluntarily submitted the issue to the trial court and already presented their evidence regarding the issue of heirship, which the RTC had consequently adjudged; or when a special proceeding had been instituted but had been finally closed and terminated, and hence, cannot be reopened.[31] Verily, we find no reason to deviate from the factual findings of the RTC on respondents' status as Adalia's legal heirs, which was proved by the evidence on record.[32]

In any event, aside from being unfounded, petitioners' argument that Demy was not Emerenciano and Adalia's natural child but merely adopted is inconsequential. Whether a natural child or adopted, like Demy, remains to be Adalia's legal heir under the law.[33]
 
The EJSW and Deed of Waiver, Quitclaim and Transfer of Residential Buildings are void insofar as respondents are concerned
 

It is undisputed that the properties subject of the EJSW and Deed of Waiver, Quitclaim and Transfer of Residential Buildings formed part of the conjugal properties of Emerenciano and Adalia. Under the regime of conjugal partnership of gains, the spouses are co-owners of all the conjugal properties.[34] Thus, when the property relation was dissolved upon Adalia's death in 1996,[35] Emerenciano, as the surviving spouse, has an actual and vested one-half undivided share in the properties.[36] The other half of the undivided share pertains to Adalia's estate, which Emerenciano and respondents, as Adalia's legal heirs, shall then co-own in equal shares pursuant to Article 980,[37] in relation to Article 979,[38] and Article 996[39] of the New Civil Code.

In the EJSW, however, Emerenciano and his minor children misrepresented that they are the sole legal heirs of Adalia[40] when the law on intestate succession does not grant any successional right from the deceased spouse to the surviving spouse's second family.[41] By such misrepresentation, respondents were unlawfully excluded from the settlement of their mother's estate. In this regard, Rule 74, Section 1 of the Rules of Court provides that "no extrajudicial settlement shall be binding upon any person who has not participated therein or had no notice thereof." Hence, we have consistently ruled that an extrajudicial settlement which excluded co-heirs of their rightful share in the inheritance is void and inexistent for having a purpose or object that is contrary to law.[42] It produces no effect whatsoever either against or in favor of anyone.[43]

Despite nullity of the extrajudicial settlement, however, the RTC aptly recognized the rights of Francis, Catherine, and Lorence over the free portion of Emerenciano's share in the properties.[44] Simply put, the conveyances under the EJSW and the Deed of Waiver, Quitclaim and Transfer of Residential Buildings in favor of Francis, Catherine, and Lorence are not totally void. We stress that the properties in question are co-owned by Emerenciano and respondents until liquidation of the conjugal partnership and proper settlement and partition of Adalia's estate.[45] In this regard, Article 493 of the Civil Code on co-ownership provides:
ART. 493. Each co-owner shall have the full ownership of his part and of the fruits and benefits pertaining thereto, and he may therefore alienate, assign or mortgage it, and even substitute another person in its enjoyment, except when personal rights are involved. But the effect of the alienation or the mortgage, with respect to the co-owners, shall be limited to the portion which may be allotted to him in the division upon the termination of the co-ownership. (Emphasis supplied)
Verily, Emerenciano's full ownership over his undivided share in the properties cannot be disregarded.[46] The conveyances under the EJSW and Deed of Waiver, Quitclaim and Transfer may be sustained to the extent of Emerenciano's undivided interest (one-half portion of the entire properties as his conjugal share[47] and one-fifth portion of the other half pertaining to Adalia's estate as his inheritance),[48] subject to proper liquidation of the conjugal partnership and partition of Adalia's estate. In addition, in view of Emerenciano's death, the Court finds it imperative to remind the parties that the conveyances of Emereciano's share made during his lifetime are further subject to the determination of the legitime of all petitioners and respondents as Emerenciano's compulsory heirs[49] in the settlement of Emerenciano's estate.[50]

From the foregoing, the RTC, as affirmed by the CA, correctly nullified the EJSW and Deed of Waiver, Quitclaim and Transfer of Residential Buildings, but only as regards the defective extrajudicial settlement and the conveyances pertaining to respondents' rightful shares, thereby making Francis, Catherine, and Lorence co-owners of the properties with respondents. This ruling conforms with Article 105[51] of the Family Code which recognizes vested rights acquired in accordance with the Civil Code or other laws in dealing with the termination of the conjugal partnership.[52] This is likewise consistent with the Court's adherence to the principle of recognizing the binding force of a contract as far as it is legally possible to do so. Quando res non valet ut ago, valeat quantum valere potest.[53]

At this juncture, it is important to emphasize that the conjugal properties are yet to be properly liquidated and partitioned. Petitioners, however, claim that respondents had received other properties from Emerenciano, and as such, had already received their rightful shares. But no evidence was presented to corroborate this claim. While properties may have been given to respondents, there was no evidence to ascertain that such properties pertain to their rightful shares in Adalia's estate, as well as in Emerenciano's estate. Pending proper liquidation and partition, it is premature to decide with specificity and finality the validity of the conveyances and the extent of their effect to respondents' interest.[54] Hence, in the interim, Francis, Catherine, and Lorence would act as trustees for the benefit of respondents with respect to any portion that might not be validly transferred and/or might belong to respondents' shares after liquidation and partition.[55] The ruling adopted in the case of Heirs of Protacio Go, Sr. v. Servacio[56] is instructive:
[I]f it turns out that the property alienated x x x really would pertain to the share of the surviving spouse, then said transaction is valid. If it turns out that there really would be, after liquidation, no more conjugal assets then the whole transaction is null and void. But if it turns out that half of the property thus alienated or mortgaged belongs to the husband as his share in the conjugal partnership, and half should go to the estate of the wife, then that corresponding to the husband is valid, and that corresponding to the other is not. Since all these can be determined only at the time the liquidation is over, it follows logically that a disposal made by the surviving spouse is not void ab initio. Thus, it has been held that the sale of conjugal properties cannot be made by the surviving spouse without the legal requirements. The sale is void as to the share of the deceased spouse (except of course as to that portion of the husband's share inherited by her as the surviving spouse). The buyers of the property that could not be validly sold become trustees of said portion for the benefit of the husband's other heirs, the cestui que trust ent. Said heirs shall not be barred by prescription or by laches (See Cuison, et al. v. Fernandez, et al., L-11764, Jan. 31, 1959.)[57]
As no definite portion may be adjudicated to any of the parties at this point, the courts a quo correctly ordered the cancellation of TCT No. T-15438, TD No. AB00303343, TD No. AB00303344, TD No. AB00303331, TD No. AB00303332, and TD No. AB00303333, which were issued by virtue of the defective EJSW and Deed of Waiver, Quitclaim and Transfer of Residential Properties, and their reversion to the originals. Anent TCT No. T-15439, which was erroneously ordered cancelled with regard only to the one-half portion of the land that it covers, should be wholly cancelled and reverted to TCT No. T-3337 for the same reason.
 
Laches and prescription cannot bar respondents' cause of action
 

Well-settled is the rule that laches and prescription cannot work against coheirs who were deprived of their lawful participation in the subject estate.[58] As found by the courts a quo, respondents had no knowledge of the questioned documents until their father's death. Upon learning that their succession rights were prejudiced, they sought annulment of the deeds within a reasonable time,[59] negating laches on their part. Anent prescription, Article 1410 of the Civil Code expressly provides that an "action or defense for the declaration of the inexistence of a contract does not prescribe."[60]

ACCORDINGLY, the Petition for Review on Certiorari is PARTLY GRANTED. The Decision dated December 4, 2020 and Resolution dated November 26, 2021 of the Court of Appeals in CA-G.R. CV No. 111116 are AFFIRMED with MODIFICATION as follows:
Declaring the Extrajudicial Settlement of Estate with Waiver VOID only insofar as the settlement is concerned;

Declaring the Extrajudicial Settlement of Estate with Waiver VALID only insofar as the conveyances pertaining to Emerenciano Delos Santos' rightful share in the properties is concerned, subject to the liquidation of the conjugal partnership and the settlement and partition of the estates of Adalia Delos Santos and Emerenciano Delos Santos with the full participation of all heirs;

Directing the Register of Deeds of Olongapo City to cancel Transfer Certificate of Title No. T-15438 and revert it to Original Certificate of Title No. P-1850, and to cancel Transfer Certificate of Title No. T-15439 in its entirety and revert it to Transfer Certificate of Title No. T-3337; and

Directing the Office of the City Assessor of Olongapo City to: (a) Cancel Tax Declaration No. AB00303343 and revert it to Tax Declaration No. AB00302198; (b) Cancel Tax Declaration No. AB00303344 and revert it to Tax Declaration. No. AB00301666; (c) Cancel Tax Declaration No. AB00303331 and revert it to Tax Declaration No. AB00302199; (d) Cancel Tax Declaration No. AB00303332 and revert it to Tax Declaration No. AB00301667; and (e) Cancel Tax Declaration No. AB00303333 and revert it to Tax Declaration No. AB00301668.
SO ORDERED.

Leonen, SAJ. (Chairperson), Lazaro-Javier, J. Lopez, and Kho, Jr., JJ., concur.

[1] The Heirs of Protacio Go, Sr. v. Servacio, 672 Phil. 447, 458 (2011) [Per J. Bersamin, First Division].

[2] See Constantino v. Heirs of Pedro Constantino, Jr., 718 Phil. 575, 594 (2013) [Per J. Perez Second Division]. (Citation omitted)

[3] Navarro v. Harris, G.R. No. 228854, March 17, 2021 [Per J. Inting, Third Division].

[4] Rollo, pp. 13-51.

[5] Penned by Associate Justice Geraldine C. Fiel-Macaraig, with the concurrence of Associate Justices Danton Q. Bueser and Florencio M. Mamauag, Jr.; id. at 57-76.

[6] Penned by Associate Justice Geraldine C. Piel-Macaraig, with the concurrence of Associate Justices Florencio Mallanao Mamauag, Jr. and Alfredo D. Ampuan; id. at 78-80.

[7] Penned by Presiding Judge Richard A. Paradeza; id. at 161-183.

[8] Id. at 103-106.

[9] Id. at 107.

[10] Id. at 108.

[11] Id. at 109-112.

[12] Id. at 113.

[13] Id. at 114.

[14] Id. at 115.

[15] Id. at 58.

[16] Id. at 116-117.

[17] Also referred to as "Lawrence" in that Deed of Waiver, Quitclaim and Transfer of Residential Buildings; id. at 118.

[18] (1) TD No. AB00303344 was issued in lieu of TD No. AB00301666; (2) TD No. AB00303331 was issued in lieu of TD No. AB00302199; (3) TD No. AB00303332 in lieu of TD No. AB00301667; and (4) TD No. AB00303333 in lieu of TD No. AB00301668; id. at 163-164.

[19] Id. at 82-88.

[20] Supra.

[21] Id. at 182-183.

[22] Supra note 5.

[23] Supra note 6.

[24] Id. at 36-41.

[25] Id. at 32-36 and 41-44.

[26] Id. at 45-49.

[27] Id. at 49-51.

[28] See Treyes v. Larlar, G.R. No. 232579, September 8, 2020 [Per J. Caguioa, En Banc].

[29] See NEW CIVIL CODE, Art. 979. Legitimate children and their descendants succeed the parents and other ascendants, without distinction as to sex or age, and even if they should come from different marriages.

An adopted child succeeds to the property of the adopting parents in the same manner as a legitimate child.

Art. 341. The adoption shall:

x x x x

(3) Make the adopted person a legal heir of the adopter.

x x x x; and

RA No. 8552. AN ACT ESTABLISHING THE RULES AND POLICIES ON THE DOMESTIC ADOPTION OF FILIPINO CHILDREN AND FOR OTHER PURPOSES, approved on February 25, 1998, Article V, Sec. 18. Succession. In legal and intestate succession, the adopter(s) and the adoptee shall have reciprocal rights of succession without distinction from legitimate filiation. x x x. (Emphasis supplied)

[30] See Treyes v. Larlar, supra note 28.

[31] Id., citing Heirs of Magdaleno Ypon v. Ricaforte, 713 Phil. 570, 576-577 (2013).

[32] Rollo, pp. 90-101.

[33] See NEW CIVIL CODE, art. 979 and 341 supra; and RA No. 8552, Article V, Section. 18 supra note 29.

[34]See NEW CIVIL CODE, art. 142. By means of the conjugal partnership of gains the husband and wife place in a common fund the fruits of their separate property and the income from their work or industry, and divide equally, upon the dissolution of the marriage or of the partnership, the net gains or benefits obtained indiscriminately by either spouse during the marriage.

[35] See FAMILY CODE, art. 126. The conjugal partnership terminates: (1) Upon the death of either spouse; x x x x.

[36] Uy v. Estate of Vipa Fernandez, 808 Phil. 470, 484 (2017) [Per J. Reyes, Third Division].

[37] Art. 980. The children of the deceased shall always inherit from him in their own right, dividing the inheritance in equal shares.

[38] Supra note 29.

[39] Art. 996. If a widow or widower and legitimate children or descendants are left, the surviving spouse has in the succession the same share as that of each of the children.

[40] Rollo, p. 116.

[41] Art. 961 of the New Civil Code enumerates those who are entitled to inheritance from a person who died intestate. It provides:

Art. 961. In default of testamentary heirs, the law vests the inheritance, in accordance with the rules hereinafter set forth, in the legitimate and illegitimate relatives of the deceased, in the surviving spouse, and in the State. (Emphasis supplied) See also Uy v. Estate of Vipa Fernandez, supra note 36.

[42] Spouses Rol v. Racho, G.R. No. 246096, January 13, 2021 [Per J. Perlas-Bernabe, Second Division]; Constantino v. Heirs of Pedro Constantino, Jr., supra note 2; and Neri v. Heirs of Hadji Usop Uy, 697 Phil. 217, 225 (2012) [Per J. Perlas-Bernabe, Second Division].

[43] Spouses Rol v. Racho, id.

[44] Rollo, p. 182.

[45] Domingo v. Spouses Molina, 785 Phil. 506, 515 (2016) [Per J. Brion, Second Division].

[46] Id.

[47] FAMILY CODE, art. 106. Under the regime of conjugal partnership of gains, the husband and wife place in a common fund the proceeds, products, fruits and income from their separate properties and those acquired by either or both spouses through their efforts or by chance, and, upon dissolution of the marriage or of the partnership, the net gains or benefits obtained by either or both spouses shall be divided equally between them, unless otherwise agreed in the marriage settlements. (Emphasis supplied)

[48] NFW CIVIL, CODE, art. 996. Supra.

[49] NEW CIVIL CODE, art. 887. The following are compulsory heirs:

(1) Legitimate children and descendants, with respect to their legitimate parents and ascendants; x x x

x x x x

(2) The widow or widower[.]

[50] See rules on legitime under Section 5 of the New Civil Code.

[51] Article 105. In case the future spouses agree in the marriage settlements that the regime or conjugal partnership of gains shall govern their property relations during marriage, the provisions in this Chapter shall be of supplementary application.

The provisions of this Chapter shall also apply to conjugal partnerships of gains already established between spouses before the effectivity of this Code, without prejudice to vested rights already acquired in accordance with the Civil Code or other laws, as provided in Article 255.

[52] The Heirs of Protacio Go, Sr. v. Servacio, 672 Phil. 447, 452 and 456-457 (2011) [Per J. Bersamin, First Division].

[53] Id.

[54] Id.

[55] Domingo v. Spouses Molina, supra note 45, citing The Heirs of Protacio Go, Sr. v. Servacio, id. at 465-466.

[56] Id.

[57] Id. at 459-460

[58] Neri v. Heirs of Hadji Yusop Uy, supra note 42 at 230.

[59] Rollo, pp. 74-75.

[60] Neri v. Heirs of Hadji Yusop Uy, supra note 42 at 230.
 
Source: Supreme Court E-Library
This page was dynamically generated
by the E-Library Content Management System (E-LibCMS)
"""
# Split text into manageable chunks (preserve structure, e.g., by paragraphs or sentences)
import re
chunks = re.split(r'(?<=\n)\n+', example.strip())  # Split by double newlines (paragraphs)

# Process each chunk and preserve all entities
results = []
for i, chunk in enumerate(chunks):
    ner_results = nlp(chunk)
    results.extend(ner_results)  # Collect all results
    
    # Temporary list to store full words with labels
    entity_words = []  
    current_word = ""
    current_label = ""
    
    for entity in ner_results:
        word = entity['word']
        label = entity['entity_group']
        
        # Combine subword tokens into one word
        if word.startswith("##"):  # Check if it’s a continuation of a word
            current_word += word[2:]  # Add the subword to the current word
        else:
            if current_word:
                entity_words.append((current_word, current_label))  # Add the previous word to the list
            current_word = word  # Start a new word
            current_label = label  # Assign the new label
    
    if current_word:  # Add the last word to the results
        entity_words.append((current_word, current_label))
    
    # Print the combined words with their labels
    for word, label in entity_words:
        print(f"Word: {word}, Label: {label}")
