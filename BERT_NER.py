from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# Initialize the pipeline with aggregation strategy
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")

# Preprocess the text
def preprocess_text(text):
    text = re.sub(r'([.,;:])', r' \1 ', text)  # Add space around punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

# Split the text into manageable chunks
def split_text(text, tokenizer, max_length=512, overlap=100):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks

# Manual chunking to handle grouped entities
def manual_chunking(ner_results):
    chunks = []
    for entity in ner_results:
        word = entity["word"]
        label = entity["entity_group"]  # Use "entity_group" instead of "entity"
        score = entity["score"] * 100  # Probability of the entity
        start = entity["start"]
        end = entity["end"]

        # Append the entity to the results
        chunks.append({"word": word, "label": label, "score": score, "start": start, "end": end})

    return chunks

# Filter entities based on context
def filter_entities(chunked_results):
    filtered_results = []
    for chunk in chunked_results:
        word = chunk["word"]
        label = chunk["label"]
        score = chunk["score"]
        if label == "PER" and len(word.split()) < 2:
            continue  # Discard single-word names
        filtered_results.append(chunk)
    return filtered_results

# Full workflow
example = """
Manila
FIRST DIVISION
[ G.R. No. 136506. January 16, 2023 ]
REPUBLIC OF THE PHILIPPINES, PETITIONER, VS. THE HONORABLE ANIANO A.
DESIERTO AS OMBUDSMAN, EDUARDO COJUANGCO, JR., JUAN PONCE ENRILE, MARIA
CLARA LOBREGAT, ROLANDO DELA CUESTA, JOSE ELEAZAR, JR., JOSE C.
CONCEPCION, DANILO URSUA, NARCISO PINEDA, AND AUGUSTO
OROSA, RESPONDENTS.
D E C I S I O N
HERNANDO,J.:
Challenged in this Petitionare the: (a) August 6, 1998 Review and Recommendationof Graft
Investigation Officer I Emora C. Pagunuran (GIO I Pagunuran), approved by Ombudsman
Aniano A. Desierto (Desierto) on August 14, 1998,dismissing petitioner Republic of the

Philippines' (Republic) Complaintfor violation of Republic Act No. (RA) 3019,docketed as OMB-
0-90-2808, filed against respondents Eduardo M. Cojuangco, Jr. (Cojuangco Jr.) Juan Ponce

Enrile (Enrile), Maria Clara Lobregat (Lobregat), Rolando Dela Cuesta (Dela Cuesta), Jose R.
Eleazar, Jr. (Eleazar, Jr.), Jose C. Concepcion (Concepcion), Danilo S. Ursua (Ursua), Narciso
M. Pineda (Pineda), and Augusto Orosa (Orosa) (collectively, respondents); and (b) GIO I
Pagunuran's September 25, 1998 Order,approved by Ombudsman Desierto on October 9, 1998,
denying petitioner Republic's subsequent Motion for Reconsiderationof the August 6, 1998
Review and Recommendation.
Procedural Antecedents
The case stemmed from the Complaintdated February 12, 1990 filed by the Office of the
Solicitor General (OSG) before the Presidential Commission on Good Government (PCGG)
against respondents Cojuangco, Jr., Enrile, Lobregat, Dela Cuesta, Eleazar, Jr., Concepcion,
Ursua, Pineda, and Orosa for violation of RA 3019 which was subsequently referred to the
Office of the Ombudsman (Ombudsman) and docketed as OMB-0-90-2808.
On August 6, 1998, GIO I Pagunuran issued a Review and Recommendationrecommending the
dismissal of the Complaint on the ground of prescription of offense.On August 14, 1998,
Ombudsman Desierto approved GIO I Pagunuran's Review and Recommendation dated August
6, 1998.Thereafter, petitioner Republic filed a Motion for Reconsiderationwhich was denied by
GIO I Pagunuran in an Order dated September 25, 1998, then approved by Ombudsman
Desierto on October 9, 1998.Hence, petitioner Republic filed a petition forcertiorariunder Rule
65 before this Court assailing the Ombudsman's dismissal of its Complaint against respondents.
On August 23, 2001, this Court grantedRepublic's petition which reversed and set aside GIO I
Pagunuran's Review and Recommendation dated August 6, 1998 and Order dated September
25, 1998, as approved by Ombudsman Desierto on August 14, 1998 and October 9, 1998,
respectively.Consequently, the Ombudsman was directed to proceed with the preliminary
investigation of OMB-0-90-2808, to wit:
WHEREFORE, the instant petition is herebyGRANTED. The assailed Review and
Recommendation dated August 6, 1998 of Graft Investigation Officer Emora C. Pagunuran, and
approved by Ombudsman Aniano A. Desierto, dismissing the petitioner's complaint in OMB-0-
90-2808, and the Order dated September 25, 1998 denying the petitioner's motion for
reconsideration, are herebyREVERSEDandSET ASIDE.
The Ombudsman is hereby directed to proceed with the preliminary investigation of the case
OMB-0-90-2808.
No pronouncement as to costs.
SO ORDERED.
Thereafter, respondents Concepcionand Lobregatfiled their Manifestations seeking to set aside
this Court's August 23, 2001 Decision on the ground of denial of due process as they were not
notified of the petition filed by Republic before this Court. In addition, respondent Cojuangco, Jr.
filed a Motion for Reconsideration of this Court's August 23, 2001 Decision.
On July 7, 2004, this Court issued a Resolution:(a) setting aside Our August 23, 2001 Decision
as the case was not yet ripe for decision; and (b) directing the petitioner Republic to serve
copies of the petition on all the respondents. As to Cojuangco, Jr.'s Motion for Reconsideration,
the same was rendered moot by this Court, to wit:

WHEREFORE, the decision of the Court dated August 23, 2001 isSET ASIDE. The petitioner is
DIRECTED to serve copies of the petition on the respondents who are directed to file their
respective Comments on the petition within ten (10) days from said service. The motion for
reconsideration of respondent Eduardo Cojuangco is MOOTED by the resolution of this Court.
SO ORDERED.
Thereafter, the Ombudsman and petitioner Republic filed their respective Motion for
Reconsideration,and Motion for Partial Reconsideration.Subsequently, this Court issued a
Resolution dated March 19, 2008 denying with finality the respective motions for lack of merit.
Hence, respondents filed their respective Comments to Republic's Petition under Rule 65,
namely: (a) respondent Concepcion's Commentdated August 27, 2004; (b) respondent Dela
Cuesta's Commentdated August 11 2008; and (c) respondents Ursua and Pineda's
Commentsdated April 13, 2009 Meanwhile, this Court noted respondent Enrile's Commentdated
May 6, 1999.
As to respondents Eleazar, Jr., Orosa, Lobregat, and Cojuangco, Jr., they were excluded as
respondents of this criminal case in view of their deaths on December 10, 2000, September 18,
2002, January 2, 2004, and June 16, 2020, respectively.The demise of respondents Eleazar, Jr.,
Orosa, Lobregat, and Cojuangco, Jr. prior to final judgment terminates their criminal

liabilitywithout prejudice to the right of the State to recover unlawfully acquired properties or ill-
gotten wealth, if any.

Background of the Case
Sometime in 1972, Agricultural Investors, Inc. (AII), a private corporation owned and/or
controlled by respondent Cojuangco, Jr., allegedly started developing a coconut seed garden in
Bugsuk Island, Palawan.
Thereafter, on November 14, 1974, then President Ferdinand E. Marcos issued Presidential
Decree No. (PD) 582,which created the Coconut Industry Development Fund (CIDF). CIDF is a
permanent fund which shall be deposited with, and administered and utilized by the Philippine
National Bank (PNB) through its subsidiary, the National Investment and Development
Corporation (NIDC), with the following purposes:
a) To finance the establishment, operation and maintenance of a hybrid coconut seednut farm
under such terms and conditions that may be negotiated by the National Investment and
Development Corporation with any private person, corporation, firm or entity as would insure
that the country shall have, at the earliest possible time, a proper, adequate and continuous
supply of high yielding hybrid seednuts;
b) To purchase all of the seednuts produced by the hybrid coconut seednut farm which shall be
distributed, for free, by the Authority to coconut farmers in accordance with, and in the manner
prescribed in, the nationwide coconut replanting program that it shall devise and
implement;Provided,That farmers who have been paying the levy herein authorized shall be
given priority;
c) To finance the establishment, operation and maintenance of extension services, model
plantations and other activities as would insure that the coconut farmers shall be informed of the
proper methods of replanting their farms with the hybrid seednuts.
"The CIDF was envisioned to finance a nationwide coconut-replanting program using
'precocious high-yielding hybrid seednuts' to be distributed for free to coconut farmers. Its initial
capital of PHP 100,000,000.00 was to be paid from the Coconut Consumers Stabilization Fund
(CCSF), with an additional amount of at least P0.20 per kilogram of copra resecada out of the
CCSF collected by the Philippine Coconut Authority."
On November 20, 1974 or six days after the creation of CIDF, NIDC accepted AII's offer, and
contracted the latter's services to implement the vital purpose of PD 582, i.e., to produce

precocious high-yielding hybrid seednuts.Thus, NIDC, represented by its then Senior Vice-
President, respondent Orosa, and AII, represented by its then Chairman and President,

respondent Cojuangco, Jr., entered into and executed a Memorandum of Agreement (MOA)on
November 20, 1974. A series of supplemental agreements and amendments subsequent to the
MOA dated November 20, 1974 were likewise executed on June 27, 1975, September 10, 1977,
April 12, 1979, and September 18, 1980, respectively.
The MOA dated November 20, 1974 principally provides that AII shall develop its coconut seed
garden in Bugsuk Island, Palawan to produce high-yielding hybrid seednuts, and thereafter, sell
its entire produce to NIDC.On the other hand, NIDC obligated itself to pay AII the cost of the

establishment, operation and maintenance of the seed garden, and support facilities; and to buy
AII's entire production of high-yielding hybrid seednuts.
However, on June 11, 1978, then President Marcos issued PD 1468,otherwise known as
theRevised Coconut Industry Code, which created the Philippine Coconut Authority (PCA). PCA
was tasked to implement and attain the State's policy "to promote the rapid integrated
development and growth of the coconut and other palm oil industry in all its aspects, and to
ensure that the coconut farmers become direct participants in, and beneficiaries of, such
development and growth."
As per Article III, Section 3 of PD 1468, the CIDF shall be administered and utilized by the bank
acquired for the benefit of the coconut farmers under PD 755Correspondingly, United Coconut
Planters Bank (UCPB), a commercial bank whose then President was respondent Cojuangco,
Jr., was acquired by the government through the CCSF for the benefit of the coconut farmers.As
a result, NIDC was substituted by UCPB as the administrator-trustee of the CIDF, and as a party
to the MOA dated November 20, 1974 with AII, and its supplements and amendments.
However, on August 27, 1982, President Marcos lifted the CCSF levy which resulted in the
depletion of the CIDF.With no financial source UCPB terminated its MOA dated November 20,
1974 with AII, and its supplements and amendments, effective December 31, 1982.
Aggrieved, AII demanded arbitration as per the arbitration clause provided in the MOA dated
November 20, 1974.Accordingly, the Board of Arbitrators (BOA), composed of Atty. Esteban
Bautista, Atty. Aniceto Dideles, and Atty. Bartolome Carale, was created to settle AII and UCPB's
obligations by reason of the termination of the MOA dated November 20, 1974, and its
supplements and amendments.
On March 29, 1983, the BOA rendered its Decision in favor of AII and awarded the latter
liquidated damages amounting to PHP 958,650,000.00 from the CIDF. "From this award was
deducted the [amount of PHP 426,261,640.00advanced by the NIDC for the development of the
seed garden, leaving a balance due to AII amounting to [PHP 532,388,354.00."In addition, the
BOA ordered that the costs of arbitration and the arbitrator's fee of PHP 150,000.00 be paid out
from the CIDF.
"On April 19, 1983, the UCPB Board of Directors, composed of respondents Cojuangco, Jr. as
President, Enrile as Chairman, Dela Cuesta, Zayco, Ursua and Pineda as members, adopted
Resolution No. 111-83, resolving to 'note' the decision of the Board of Arbitrators, allowing the
arbitral award to lapse with finality."
Thereafter, on February 12, 1990, petitioner Republic filed the subject Complaint against
respondents Cojuangco, Jr., Enrile, Lobregat, Dela Cuesta, Eleazar, Jr., Concepcion, Ursua,
Pineda and Orosa.
Petitioner Republic's Complaint (OMB-0-90-2808)
Republic averred that respondent Cojuangco, Jr. took advantage of his close relationship with
then President Marcos for his own personal and business interests through the issuance of
favorable decrees.Cojuangco, Jr. caused the Philippine Government, through the NIDC, to enter
into a contract with him, through its corporation AII, under terms and conditions grossly
disadvantageous to the government and in conspiracy with the members of the UCPB Board of
Directors, in flagrant breach of fiduciary duty as administrator-trustee of the CIDF.
Specifically, petitioner Republic averred that the MOA dated November 20, 1974 is a one-sided
contract with provisions clearly in favor of AII, and thereby allegedly placed NIDC in a no-win
situation.Petitioner cited several stipulations in the MOA dated November 20, 1974 to
substantiate its claim, to wit:
1. Under Section 9.1 of the MOA, neither party shall be liable for any loss or damage due to the
non-performance of their respective obligations resulting from any cause beyond the reasonable
control of the party concerned. However, under Section 9.3, notwithstanding the occurrence of
such causes, the obligation of the NIDC to pay AII's share of the development costs amounting
to PHP 426,260,000.00 would still remain enforceable.£A⩊phi
2. Under Sec. 11.2, if NIDC fails to perform its obligations, for any cause whatsoever, it will be
liable out of the CIDF, not only for the development costs, but also for liquidated damages equal
to the stipulated price of the hybrid seednuts for a period of five years at the rate of 19,173,000
seednutsper annum, totaling PHP 958,650.00.
3. Under Section 11.3, while AII was given the right to terminate the contract in case offorce
majeure, no such right was given in favor of NIDC. Moreover, AII can do so without incurring any

liability for damages.
4. AII was only required to exert best efforts to produce a projected number of seednuts while
NIDC was required to set aside and reserve from CIDF such amount as would insure full and
prompt payment.
As to respondents Enrile, Dela Cuesta, Concepcion, Ursua, and Pineda, petitioner Republic
averred that as members of the UCPB Board of Directors, their act of allowing the BOA's March
29, 1983's Decision to lapse into finality resulted in the successful siphoning of PHP
840,789,855.33 from CIDF to AII a corporation owned by respondent Cojuangco, Jr. Thus,
respondents as members and officers of UCPB, a government-owned and controlled
corporation having been acquired by the government through the CCSF levy, are considered
public officers within the contemplation of RA 3019. Furthermore, respondent Cojuangco, Jr. is
considered a public officer being then the Director of PCA, the President of UCPB, and
ambassador-at-large.
Petitioner Republic further noted respondents Enrile, Cojuangco, Jr., Dela Cuesta, and
Concepcion's respective positions as Chairman/Director, President/Director, Corporate
Secretary, and Treasurer/Director of AII until their resignation on November 8, 1982, as well as
respondent Orosa's participation as Senior Vice-President of NIDC and the latter's
representative to the execution of the MOA dated November 20, 1974.
Petitioner Republic essentially professed that respondents are directly or indirectly interested in
personal gain, or had material interest in the transaction requiring the approval of a board,
panel, or group in which they were members, in violation of RA 3019 to the grave damage and
prejudice of the public interest, the Filipino people, the Republic, and the coconut farmers.
Decision of the Ombudsman (OMB-0-90-2808)
On August 6, 1998, GIO I Pagunuran issued a Review and Recommendation, which was
approved by Ombudsman Desierto on August 14, 1998, dismissing petitioner Republic's
Complaint against respondents on the ground of prescription.The Ombudsman reckoned the
prescriptive period from the execution of the MOA on November 20, 1974. Since the case was
filed only on February 12, 1990, the Ombudsman ruled that the same was filed beyond the
prescriptive period of 10 years under Sec. 11 of RA 3019.Also, the Ombudsman declared that
the MOA dated November 20, 1974 was confirmed and ratified by PD 961 and PD 1468 and
therefore, was given legislative imprimatur, to wit:
It appears, therefore, that the execution of the questioned contracts and substitution of the NIDC
by the UCPB were given legislative imprimatur. The ratification of the question[ed] MOA, its
amendments and supplements by P.D. Nos. 961 and 1468 was, at the very least, a declaration
on the part of the government that the questioned contracts are, in fact, valid, legal and
beneficial to the government and the Republic and that the act of the officers of the NIDC of
entering into the questioned contracts were, in fact valid and legal. The said laws have not been
repealed nor declared constitutional and, therefore, remain valid and effective to
date. Respondents, are therefore, protected by the mantle of legality which all valid laws cast
upon those who abide by them.
WHEREFORE, premises considered, it is respectfully recommended that the complaint be, as it
is hereby, dismissed.(Emphasis supplied)
Petitioner Republic filed its Motion for Reconsiderationon the following grounds: (a) the offense
charged in the Complaint falls within the category of an ill-gotten wealth case, which under the
Constitution is imprescriptible; and (b) void contracts are not subject to ratification and/or
confirmation. However, the said motion was denied by GIO I Pagunuran in the Order dated
September 25, 1998, which was approved by Ombudsman Desierto on October 9, 1998.
Hence, this Petition under Rule 65.
Issues
Petitioner Republic presented the following issues for Our resolution:
I.
WHETHER THE OMBUDSMAN ACTED WITH GRAVE ABUSE OF DISCRETION IN
DECLARING THAT THE OFFENSE CHARGED IN THE COMPLAINT FOR VIOLATION OF R.A.
NO. 3019 HAD ALREADY PRESCRIBED WHEN THE COMPLAINT WAS FILED.
II.
WHETHER THE OMBUDSMAN ACTED WITH GRAVE ABUSE OF DISCRETION IN
DECLARING THAT THERE IS NO BASIS TO INDICT PRIVATE RESPONDENTS FOR

VIOLATION OF THE ANTI-GRAFT LAW BASED ON THE CONTRACT IN QUESTION.
The Petition
Petitioner Republic opines that although the complaint filed against respondents is for violation
of RA 3019, the same is related to the efforts of the government to recover ill-gotten wealth from
President Marcos' cronies, or from persons closely and personally associated with him. Thus,
the right of the State to recover such properties unlawfully acquired by public officials or
employees shall not be barred by prescription, laches, or estoppel as per Sec. 15, Art. XI of the
1987 Constitution.
Furthermore, the Constitutional Commission's intention in drafting Sec. 15, Art. XI of the 1987
Constitution was to make imprescriptibility applicable to both civil and criminal aspects of the
case. Thus, while the phrase "or to prosecute offenses in connection therewith" was omitted or
deleted in the final version of Sec. 15, Art. XI, such omission or deletion should not override the
manifest intent of the Constitutional Commission to make the prosecution of offenses related to
ill-gotten wealth imprescriptible.
At the time of the execution of the MOA on November 20, 1974 the period of prescription under
Sec. 11 of RA 3019 was 10 years. However, on March 16, 1982, Batas Pambansa Bilang (BP)
195 was enacted amending the prescriptive period of violation of RA 3019 to 15 years. Thus, at
the time of the adoption of the 1987 Constitution, the period of prescription for violation of RA
3019 reckoned from the execution of the MOA on November 20, 1974 has not yet set in. Also,
Sec. 11 of RA 3019 was similarly amended by Sec. 15, Art. XI of the 1987 Constitution with
respect to imprescriptibility of offenses related to ill-gotten wealth.
Nonetheless, granting that the offense committed by respondents is not imprescriptible, the
reckoning point shall not be from the execution of the MOA on November 20, 1974, but from the
EDSA Revolution in February 1986. The Republic, citing Sec. 2 of Act No. 3326,which provides
that "prescription shall begin to run from the day of the commission of the violation of law, and if
the same be not known at the time, from the discovery thereof and the institution of judicial
proceedings," argues that the reckoning point of the prescriptive period must be from the
discovery of the alleged violation, i.e., at the time of the EDSA Revolution in February 1986 and
not from the execution of the MOA on November 20, 1974.
Republic explains that the acts complained of were committed during the Marcos regime by
persons closely associated with President Marcos, which means that no one could have known
the existence of the said MOA dated November 20, 1974 except respondents themselves. Even
assuming that third parties knew of the existence of the subject MOA, no one had the
reasonable opportunity nor political will to prosecute respondents or the persons involved
therein. Considering the peculiar circumstances at that time, the prescriptive period should be
reckoned from the discovery of the offense, i.e., immediately after the EDSA Revolution in
February 1986. Thus, since the complaint was only filed in 1990 or four years from 1986, the
offense charged against respondents has not yet prescribed.
In addition, the Ombudsman's ruling that the subject MOA is not grossly and manifestly
disadvantageous to the government, as it was confirmed and ratified by PD 582 and PD 1468, is
not correct. The Ombudsman's conclusion: (a) that the ratification of the subject MOA is at the
very least a declaration on the part of the government that the agreement is valid, legal, and
beneficial to it; and (b) that the act of the officers of NIDC were valid and legal, is palpable error
as it gives upon the lawmaker the power to adjudicate on the validity of contracts which is
essentially a judicial function.
Also, the terms and conditions of the subject MOA were on its face grossly and manifestly
disadvantageous to the government. The Republic likewise assails the failure of the UCPB BOD
to appeal and question the arbitral award in favor of AII to the detriment of the CIDF.
Respondent Enrile's Arguments
Respondent Enrile contends that Republic's Petition under Rule 65 should be dismissed as it is
a mere attempt to substitute a lost appeal. He notes that Sec. 27 of RA 6770,otherwise known
as theOmbudsman Act, provides for the specific mode or manner of assailing orders, directives
or decisions issued by the Office of the Ombudsman, specifically, by filing a petition
forcertiorariwithin 10 days from receipt of the written notice of the order, directive or decision, or
denial of the motion for reconsideration in accordance with Rule 45 of the Rules of Court.
Thus, the Republic chose the wrong mode of appeal when it filed a petition under Rule 65. By
filing a petition under Rule 65 instead of Rule 45, the Republic has clearly failed to avail of the

plain, speedy, and adequate remedy provided by law within the reglementary period, i.e., Sec.
27 of RA 6770 or a petition under Rule 45. Considering the Republic's failure to file a petition
under Rule 45 within 10 days from receipt of the Ombudsman's denial of its motion for
reconsideration, it lost the right to assail both the Review and Recommendation dated August 6,
1998 and Order dated September 25, 1998.
Even assuming that the petition under Rule 65 is the correct remedy, the same was filed out of
time which warrants its outright dismissal. The Republic had 60 days from the receipt of the
Review of Recommendation dated August 6, 1998 as approved by Ombudsman
Desierto,i.e.,from August 28, 1998, within which to file a petition under Rule 65. However, it filed
a motion for reconsideration on September 11, 1998 instead which effectively interrupts the filing
of the petition under Rule 65. As per Sec. 4, Rule 65, the petitioner has only the balance of the
60-day period from the notice of the denial of the motion for reconsideration within which to file a
petition under Rule 65 or until December 13, 1998. Hence, Republic's filing of the instant petition
only on December 28, 1998 is clearly beyond the reglementary period.
Also, the offense charged in OMB-0-9-2808 has already prescribed Sec. 15, Art. XI of the 1987
Constitution cannot be given an amending or repealing effect as it would impair vested rights.
When the complaint was filed, he had already acquired a vested right to be protected on the
ground of prescription of the offense charged against him.
Moreover, the strict application of Sec. 15, Art. XI,i.e.,imprescriptibility of criminal offense in
relation to ill-gotten wealth, will violate Sec. 22, Art. III of the Constitution which prohibits
enactment of bills of attainder andex-post factolaws. Even if the prescriptive period is 15 years
as per BP 195, the offense has already prescribed since the running of the prescriptive period
has not been interrupted by any judicial proceeding, i.e., filing of the criminal case in court, as
required under Sec. 2 of Act No. 3326.
He insists that the running of the prescriptive period should be reckoned from the execution of
the MOA on November 20, 1974. The MOA is duly notarized which makes it a public document
subject to examination and discovery of anyone. During the execution of the MOA, civil courts
were open and functioning which negates Republic's contention that there is no available forum
to question the legality of the agreement.
Furthermore, not only did PD 961 and PD 1468 confirm and ratify the MOA dated November 20,
1974, and its supplements and amendments, they also elevated the MOA from a mere
agreement binding only between parties to a contract with force and effect of law. Thus, even if
Enrile participated in the negotiation, perfection, and enforcement of the MOA dated November
20, 1974, he cannot be made criminally liable because his acts and/or involvement therein were
mandated by law.
Lastly, the terms and conditions of the MOA dated November 20, 1974 are fair and reasonable
to both parties and are not disadvantageous to the government. No liability may be imputed to
him in merely noting the subject arbitral award in favor of AII instead of assailing the same.
When he signed Resolution No. 111-83, it was for the sole purpose of attesting to the truth and
correctness of the minutes and not for the purpose of approving any action or resolution. He
also stresses that UCPB is a commercial and private bank owned by coconut farmers and not
by the government. Thus, he cannot be made liable as a public officer as defined under Sec. 2
of RA 3019 for his acts as the Chairman of UCPB's BOD.Respondent Concepcion's Arguments
Respondent Concepcion argues that the petition should be dismissed for failure of the Republic
to comply with Sec. 3, Rule 46, that is, the petition shall be filed with proof of service thereof on
the respondent. In fact, Republic admitted that it had not served copies of the petition or
subsequent pleadings to respondents Concepcion and Lobregat. Moreover, respondent
Concepcion contends that since he was served a copy of the petition only on August 3, 2004,
the petition is deemed to have been properly filed only on such date. Thus, the petition is filed
beyond the 60-day reglementary period under Rule 65.
Further, the Republic has no cause of action against him in view of the dismissal of Civil Case
No. 0033-C filed before the Sandiganbayan involving the same acts or omissions as in the
present case. The quantum of proof required in civil cases is preponderance of evidence while
in criminal cases, as in the case at bar, is proof beyond reasonable doubt. Hence, with the
dismissal of SB Civil Case No. 0033-C against him, which only requires a lesser quantum of
proof, there is more reason to dismiss the criminal case OMB-0-9-2808 filed against respondent
Concepcion.

In addition, his participation in the alleged acts or omissions subject of this criminal case OMB-
0-9-2808 was done and/or performed in the course of his professional duties as a lawyer. He

citesRegala v. Sandiganbayan(Regala) wherein this Court declared that the Angara Abello
Concepcion Regala & Cruz (ACCRA) lawyers, of which respondent Concepcion is a partner,
were being prosecuted solely on the basis of activities and services performed in the course of
their duties as lawyers. Hence, the PCGG had no valid cause of action against them, and the
ACCRA lawyers were excluded as parties-defendants in SB Civil Case No. 0033-C. Similarly,
respondent Concepcion prays for the dismissal of criminal case OMB-0-90-2808 filed against
him as there is no reason to prosecute him as a lawyer.
Moreover, the running of the 10-year prescriptive period should be reckoned from the EDSA
Revolution in 1986. Thus, the offense shall prescribe in 1996. In order to interrupt the
prescription of offense, a criminal proceeding should be instituted before the Sandiganbayan.
However, no criminal proceedings have been instituted against respondent Concepcion before
the lapse of the prescriptive period in 1996. Hence, he can no longer be validly prosecuted for
the acts or omissions subject of OMB-0-90-2808.
Nonetheless, inPresidential Ad Hoc Fact-Finding Committee on Behest Loans v. Desierto,this
Court ruled that the prescriptive period was interrupted upon the filing of the complaint with the
Ombudsman. Respondent Concepcion however, contends that this new doctrine should be
prospectively applied considering that it was rendered by this Court years after the alleged
commission of the subject offense charged and after the filing of OMB-0-90-2808. Thus, to
retroactively apply the said ruling would constitutionally impair his substantive right to
prescription.Respondent Dela Cuesta's Arguments
Similarly, respondent Dela Cuesta opines that due to Republic's failure to appeal within the
reglementary period, the Ombudsman's Review and Recommendation dated August 6, 1998
became final and executory. Even assuming that the correct remedy is a petition
forcertiorariunder Rule 65, the same was filed out of time. Dela Cuesta notes that Republic only
served him a copy of the subject Petition on June 27, 2008 or almost nine years and eight
months from the time the Republic was notified of the Order dated September 25, 1998. Thus,
the petition should be dismissed for being invalid and contrary to Sec. 3, Rule 46 of the Rules of
Court.
In the same vein, the time to prosecute the offense charged has already prescribed. The alleged
violation of RA 3019 was committed on November 20, 1974. Thus, since no judicial proceedings
are instituted against respondents, the running of the prescriptive period has not been
interrupted.
He likewise argues that the MOA dated November 20, 1974 is duly notarized which makes it a
public document subject to the examination and discovery of any one with the exercise of
reasonable diligence. At the time of the execution of the MOA, civil courts are open and
functioning. In addition, PD 961 and PD 1468 ratified the subject MOA which negates Republic's
allegation that the terms of the MOA are grossly disadvantageous to the government.
Even granting that the reckoning point of the prescriptive period, i.e., 15 years, is in February
1986, the offense charged has still prescribed since he only became a party to the instant
petition on June 27, 2008 or three years and four months after the prescription of offense on
February 12, 2005.
Furthermore, the petition should be dismissed for palpable violations of his right to speedy
disposition of cases. A period of 17 years reckoned from the filing of the case before the PCGG
is a long period of time. Also, the reasons for the delay are all attributable to the Republic. He
further contends that this is the only opportune time for him to invoke his right to speedy
disposition as he was not served a copy of the instant petition earlier. He adds that he invoked
his right to speedy disposition in related cases filed against him.
Lastly, Dela Cuesta opines that Civil Case No. 0033-C, which pertains to the same act or
omission alleged in the case at bar, pending before the Sandiganbayan, excluded him as a
respondent. He argues that since the quantum of evidence required in civil cases is only
preponderance of evidence, the subject criminal case with a higher quantum of evidence cannot
prosper.Respondents Ursuaand Pineda'sArguments
Respondents Ursua and Pineda contend that other than self-serving statements, petitioner
Republic failed to offer any substantial evidence to support the alleged grave abuse of
discretion, manifest partiality, evident bad faith, or inexcusable negligence on the part of the

Ombudsman. They citedEspinosa v. Office of the OmbudsmanandThe Presidential Ad-Hoc Fact
Finding Committee on Behest Loans v. Desiertoto support their contention that courts
consistently refrain from interfering with the Ombudsman's powers and independence.
Lastly, citingPresidential Commission on Good Government v. Desierto,they argue that the
Ombudsman has the power to determine whether there exists reasonable ground to believe that
a crime has been committed and that the accused is probably guilty thereof. The Court will not
ordinarily interfere with the Ombudsman's exercise of its investigatory and prosecutorial powers
without good and compelling reasons. While there are certain exceptions when the Court may
intervene, respondents Ursua and Pineda aver that none applies in the present case.
Our Ruling
We find the petition partly meritorious.
Death of accused during the pendency of a criminal action
At the outset, respondents Eleazar, Jr., Orosa, Lobregat, and Cojuangco Jr died during the
pendency of this petition. Article 89 of the Revised Penal Code states that:
ART. 89.How criminal liability is totally extinguished. Criminal liability is totally extinguished
1. By the death of the convict, as to the personal penalties; and as to pecuniary penalties,
liability therefor is extinguished only when the death of the offender occurs before final
judgment[.]
InPeople v. Bayotas,We explained the effects of the death of the accused pending appeal, to
wit:
1. Death of the accused pending appeal of his/[her] conviction extinguishes his/[her] criminal
liability as well as the civil liability based solely thereon. As opined by Justice Regalado, in this
regard, "the death of the accused prior to final judgment terminates his/[her] criminal liability
andonlythe civil liabilitydirectlyarising from and based solely on the offense committed,i.e., civil
liabilityex delicto in senso strictiore."
2. Corollarily, the claim for civil liability survives notwithstanding the death of accused, if the
same may also be predicated on a source of obligation other thandelict.Article 1157 of the Civil
Code enumerates these other sources of obligation from which the civil liability may arise as a
result of the same act or omission:
a) Law
b) Contracts
c) Quasi-contracts
d) [x x x]
e) Quasi-delicts
3. Where the civil liability survives, as explained in Number 2 above, an action for recovery
therefor may be pursued but only by way of filing a separate civil action and subject to Section
1, Rule 111 of the 1985 Rules on Criminal Procedure as amended. This separate civil action
may be enforced either against the executor/administrator or the estate of the accused,
depending on the source of obligation upon which the same is based as explained above.
4. Finally, the private offended party need not fear a forfeiture of his/[her] right to file this
separate civil action by prescription, in cases where during the prosecution of the criminal action
and prior to its extinction, the private-offended party instituted together therewith the civil action.
In such case, the statute of limitations on the civil liability is deemed interrupted during the
pendency of the criminal case, conformably with provisions of Article 1155 of the Civil Code, that
should thereby avoid any apprehension on a possible privation of right by prescription.
With the demise of respondents Eleazar, Jr., Orosa, Lobregat, and Cojuangco, Jr., their criminal
liabilities and civil liability ex delicto are now extinguished. For the civil liability, which may be
based on sources other than delict, the Republic may file a separate civil action against the
estate of respondents Eleazar, Jr., Orosa, Lobregat, and Cojuangco, Jr. as may be warranted by
law and procedural rules; or if already filed, the said separate civil action shall survive
notwithstanding the dismissal of the criminal case in view of their deaths.
Apropos,the subsequent discussion pertains only to the imputation of grave abuse of discretion
on the Ombudsman as to its order of dismissal of the Complaint against respondents Enrile,
Dela Cuesta, Concepcion, Ursua, and Pineda on the ground of prescription.
Propriety of the Petition
Before We delve into the merits of the case, We deem it necessary to determine the propriety of
the petition. Sec. 27 of RA 6770 provides that:

Effectivity and Finality of Decisions. x x
x x x x
In all administrative disciplinary cases, orders, directives, or decisions of the Office of the
Ombudsman may be appealed to the Supreme Court by filing a petition forcertiorariwithin ten
(10) days from receipt of the written notice of the order, directive or decision or denial of the
motion for reconsideration in accordance with Rule 45 of the Rules of Court. (Emphasis
supplied)
However, the above provision was already declared unconstitutional inFabian v. Desiertofor
expanding the Supreme Court's jurisdiction without its consent in violation of Art. VI, Sec. 30 of
the Constitution, to wit:
Taking all the foregoing circumstances in their true legal roles and effects, therefore, Section 27
of Republic Act No. 6770 cannot validly authorize an appeal to this Court from decisions of the
Office of the Ombudsman in administrative disciplinary cases. It consequently violates the
proscription in Section 30, Article VI of the Constitution against a law which increases
the appellate jurisdiction of this Court. No countervailing argument has been cogently presented
to justify such disregard of the constitutional prohibition which, as correctly explained inFirst
Lepanto Ceramics, Inc. vs. The Court of Appeals, et al.,was intended to give this Court a
measure of control over cases placed under its appellate jurisdiction. Otherwise, the
indiscriminate enactment of legislation enlarging its appellate jurisdiction would unnecessarily
burden the Court.(Citations omitted)
Also, Sec. 27 of RA 6770 only relates to administrative disciplinary cases.It does not apply to
appeals from Ombudsman's rulings in criminal cases,nor to resolutions on preliminary
investigationssuch as the case at bar. InNava v. Commission on Audit,We declared that the
remedy of an aggrieved party in such criminal case is an action forcertiorariunder Rule 65, to
wit:
The remedy availed of by petitioner is erroneous. Instead of a petition forcertiorariunder Rule 65
of the Rules of Court, petitioner filed with this Court the present petition for review
oncertiorariunder Rule 45 of the Rules of Court pursuant to the provisions of Section 27 of
Republic Act No. 6770.
Rule 45 of the Rules of Court provides that only judgments or final orders or resolutions of the
Court of Appeals, Sandiganbayan, the Regional Trial Court and other courts, whenever
authorized by law, may be the subject of an appeal bycertiorarito this Court. It does not include
resolutions of the Ombudsman on preliminary investigations in criminal cases. Petitioner's
reliance on Section 27 of R.A. No. 6770 is misplaced.Section 27 is involved only whenever an
appeal bycertiorariunder Rule 45 is taken from a decision in an administrative disciplinary
action. It cannot be taken into account where an original action forcertiorariunder Rule 65 is
resorted to as a remedy for judicial review, such as from an incident in a criminal action. In other
words, the right to appeal is not granted to parties aggrieved by orders and decisions of the
Ombudsman in criminal cases, like the case at bar. Such right is granted only from orders or
decisions of the Ombudsman in administrative cases.
An aggrieved party is not left without any recourse.Where the findings of the Ombudsman as to
the existence of probable cause is tainted with grave abuse of discretion amounting to lack or
excess of jurisdiction, the aggrieved party may file a petition forcertiorariunder Rule 65 of the
Rules of Court.(Citations omitted, emphases and underscoring supplied)
InTirol, Jr. v. Del Rosario,We explained that although the law is silent as to the remedy of the
aggrieved in criminal cases, the party is not without recourse as he or she can assail the
Ombudsman's finding of probable cause in a petition forcertiorariunder Rule 65 if the same is
tainted with grave abuse of discretion, amounting to lack or excess of jurisdiction,viz.:
Section 27 of R.A. No. 6770 provides that orders, directives and decisions of the Ombudsman in
administrative cases are appealable to the Supreme CourtviaRule 45 of the Rules of Court.
However, inFabian v. Desierto,we declared that Section 27 is unconstitutional since it expanded
the Supreme Court's jurisdiction, without its advice and consent, in violation of Article VI, Section
30 of the Constitution. Hence, all appeals from decisions of the Ombudsman in administrative
disciplinary cases may be taken to the Court of Appeals under Rule 43 of the 1997 Rules of Civil
Procedure.
True, the law is silent on the remedy of an aggrieved party in case the Ombudsman found
sufficient cause to indict him in criminal or non-administrative cases.We cannot supply such

deficiency if none has been provided in the law. We have held that the right to appeal is a mere
statutory privilege and may be exercised only in the manner prescribed by, and in accordance
with, the provisions of law. Hence, there must be a law expressly granting such privilege. The
Ombudsman Act specifically deals with the remedy of an aggrieved party from orders, directives
and decisions of the Ombudsman in administrative disciplinary cases. As we ruled inFabian, the
aggrieved party is given the right to appeal to the Court of Appeals. Such right of appeal is not
granted to parties aggrieved by orders and decisions of the Ombudsman in criminal cases, like
finding probable cause to indict accused persons.
However, an aggrieved party is not without recourse where the finding of the Ombudsman as to
the existence of probable cause is tainted with grave abuse of discretion, amounting to lack or
excess of jurisdiction. An aggrieved party may file a petition forcertiorariunder Rule 65 of the
1997 Rules of Civil Procedure.(Citations omitted, emphases and underscoring supplied)
Verily, petitioner Republic correctly availed of the remedy of petition forcertiorariunder Rule 65
when it assailed Ombudsman's August 6, 1998 Review and Recommendation and the
September 25, 1998 Order which dismissed the complaint against respondents for violation of
RA 3019. Petitioner Republic received a copy of the August 6, 1998 Review and
Recommendation on August 28, 1998 and the September 25, 1998 Order on October 28, 1998.
Prior to the Court's promulgation of A.M. No. 00-2-03-SC,Sec. 4, Rule 65 of the Rules of Court
provides that in case the aggrieved party's motion for new trial or reconsideration is denied, he
or she may file a petition under Rule 65 within the remaining period of 60 days, which in no case
shall be less than five days, to wit:
SEC. 4.Where and when petition to be filed. The petition may be filed not later than sixty (60
days from notice of the judgment, order or resolution sought to be assailed in the Supreme
Court or, if it relates to the acts or omissions of a lower court or of a corporation, board, officer or
person, in the Regional Trial Court exercising jurisdiction over the territorial area as defined by
the Supreme Court. It may also be filed in the Court of Appeals whether or not the same is in aid
of its appellate jurisdiction, or in the Sandiganbayan if it is in aid of its jurisdiction. If it involves
the acts or omissions of a quasi-judicial agency, and unless otherwise provided by law or these
Rules, the petition shall be filed in and cognizable only by the Court of Appeals.
If the petitioner had filed a motion for new trial or reconsideration in due time after notice of said
judgment, order or resolution, the period herein fixed shall be interrupted. If the motion is
denied, the aggrieved party may file the petition within the remaining period, but which shall not
be less than five (5) days in any event, reckoned from notice of such denial. No extension of
time to file the petition shall be granted except for the most compelling reason and in no case to
exceed fifteen (15) days.(Emphasis and underscoring supplied)
Applying the foregoing, petitioner Republic had until December 13, 1998 within which to file a
petition forcertiorariunder Rule 65. However, it only filed the instant petition on December 28,
1998 or 15 days beyond the 60-day reglementary period. Patently, petitioner Republic's petition
is filed out of time as per the above-quoted provision.
Nevertheless, during the pendency of the petition, the Court promulgated A.M. No. 00-2-03-SC,
which amended Sec. 4 of Rule 65 and became effective on September 1, 2000, to wit:
SECTION 4.When and where petition filed. The petition shall be filed not later than sixty (60
days from notice of judgment, order or resolution. In case a motion for reconsideration or new
trial is timely filed, whether such motion is required or not, the sixty (60) day period shall be
counted from notice of the denial of said motion.
The petition shall be filed in the Supreme Court or, if it relates to the acts or omissions of a lower
court or of a corporation, board, officer or person, in the Regional Trial Court exercising
jurisdiction over the territorial area as defined by the Supreme Court. It may also be filed in the
Court of Appeals whether or not the same is in aid of its appellate jurisdiction, or in the
Sandiganbayan if it is in aid of its appellate jurisdiction. If it involves the acts or omissions of a
quasi-judicial agency, unless otherwise provided by law or these rules, the petition shall be filed
in and cognizable only by the Court of Appeals.
No extension of time to file the petition shall be granted except for compelling reason and in no
case exceeding fifteen (15) days.(Emphases and underscoring supplied)
Settled is the rule that statutes regulating the procedure of the courts are construed as
applicable to actions pending and undetermined at the time of their passageSince A.M. No. 00-

2-03-SC relates to the mode of procedure, i.e., the reglementary period within which to file a
petition forcertiorariunder Rule 65, it is applicable to pending cases at the time of its adoption.
In the present case, it is apparent that the petition is still pending resolution before this Court
when A.M. No. 00-2-03-SC was issued. Similarly, the Court applied A.M. No. 00-2-03-SC
retrospectively inPresidential Commission on Good Government v. Desiertowhen it ruled that:
Prefatorily, the petition should have been dismissed for late filing. Petitioner received a copy of
the assailed resolution on 08 April 1999. A motion for reconsideration was filed by the PCGG on
12 April 1999. On 06 August 1999, it received a copy of the order denying its motion for
reconsideration. Pursuant to Section 65 of the 1997 Rules of Civil Procedure, the petition should
have been filed on 02 October 1999; instead, the petition was only posted on 05 October 1999.
During the pendency of this case, however, the Court promulgated A.M. No. 00-2-03-SC
(Further Amending Section 4, Rule 65 of the 1997 Rules on Civil Procedure), made effective on
01 September 2000, that provided:
SECTION 4.When and where petition filed. The petition shall be filed not later than sixty (60
days from notice of judgment, order or resolution. In case a motion for reconsideration or new
trial is timely filed, whether such motion is required or not, the sixty (60) day period shall be
counted from notice of the denial of said motion.
In view of the retroactive application of procedural laws, the instant petition should now be
considered timely filed.(Emphasis and underscoring supplied)
InArk Travel Express Inc. v. Abrogar,the Court upheld the retroactive application of A.M. No. 00-
2-03-SC to pending cases before it, to wit:
The issue raised in the present petition concerns the jurisdiction of the RTC in ordering the
dismissal of the criminal cases pending before the MTC and therefore, the proper remedy
iscertiorari. As such, the present petition forcertiorariought to have been dismissed for late filing.
The assailed Order dated October 2, 1998 was received by Ark Travel on October 16, 1998. Ark
Travel filed the Motion for Reconsideration fourteen days later or on October 30, 1998. On
November 27, 1998, Ark Travel received the Order of the denial of the Motion for
Reconsideration. Pursuant to Rule 65 of the 1997 Rules on Civil Procedure, then prevailing, the
petition should have been filed on the forty-sixth day (60 days minus 14 days) from November
27, 1998 or on January 12, 1999, the last day of the 60-day reglementary period; instead, the
petition was filed on January 26, 1999.
However, during the [sic] pendency of herein petition, the Court promulgated A.M. No. 00-2-03,
amending Section 4, Rule 65 of the 1997 Rules on Civil Procedure, effective September 1,
2000, to wit:
SEC. 4.When and where petition filed. The petition shall be filed not later than sixty (60) day
from notice of judgment, order or resolution. In case a motion for reconsideration or new trial is
timely filed, whether such motion is required or not, the sixty (60) day period shall be counted
from notice of the denial of said motion.
In which case, the filing of the petition on January 26, 1999 was filed on the 60th day from
November 27, 1998, Ark Travel's date of receipt of notice of the order denying Ark Travel's
motion for reconsideration.
We have consistently held that statutes regulating the procedure of the courts will be construed
as applicable to actions pending and undetermined at the time of their passage procedura
laws are retroactive in that sense and to that extent.In view of such retroactive application of
procedural laws the instant petition should be considered as timely filed.(Emphasis and
underscoring supplied)
Therefore, the retroactive application of A.M. No. 00-2-03-SC, specifically the 60-day period
within which to file a petition forcertiorari, which must be reckoned from the notice of the denial
of a motion for reconsideration or new trial, shall also be applied to the present case. Thus,
petitioner Republic had 60 days from receipt of the September 25, 1998 Order or until
December 27, 1998 within which to file a petition. However, since December 27, 1998 is a
Sunday, petitioner Republic's filing of its petition on December 28, 1998 is considered timely
filed within the 60-day reglementary period.
As to the alleged failure of the Republic to timely serve copies of the petition to respondents
Concepcion and Lobregat, the pertinent provisions of Sec. 6, Rule 65; Sec. 2, Rule 56; and Sec.
2, 3, and 4, Rule 46 of the Rules of Court are particularly instructive of the effects thereof:

Section 6, Rule 65
SECTION 6.Order to Comment. If the petition is sufficient in form and substance to justif
such process, the court shall issue an order requiring the respondent or respondents to
comment on the petition within ten (10) days from receipt of a copy thereof. Such order shall be
served on the respondents in such manner as the court may direct, together with a copy of the
petition and any annexes thereto.
In petitions forcertioraribefore the Supreme Court and the Court of Appeals, the provisions of
Section 2, Rule 56, shall be observed.Before giving due course thereto, the court may require
the respondents to file their comment to, and not a motion to dismiss, the petition. Thereafter,
the court may require the filing of a reply and such other responsive or other pleadings as it may
deem necessary and proper. (Emphasis and underscoring supplied)
Section 2, Rule 56
SECTION 2.Rules Applicable.The procedure in original case
forcertiorari, prohibition,mandamus, quo warrantoandhabeas corpusshall be in accordance with
the applicable provisions of the Constitution, laws, and Rules 46, 48, 49, 51, 52 and this
Rule,subject to the following provisions:
a) All references in said Rules to the Court of Appeals shall be understood to also apply to the
Supreme Court;
b) The portions of said Rules dealing strictly with and specifically intended for appealed cases in
the Court of Appeals shall not be applicable; and
c)Eighteen (18) clearly legible copies of the petition shall be filed, together with proof of service
on all adverse parties.(Emphases and underscoring supplied)
Sections 2, 3 and 4, Rule 46
SECTION 2.To What Actions Applicable.This Rule shall apply to original action
forcertiorari, prohibition,mandamus and quo warranto.
Except as otherwise provided, the actions for annulment of judgment shall be governed by Rule
47, forcertiorari, prohibition andmandamusby Rule 65, and for quo warranto by Rule 66.
SECTION 3.Contents and Filing of Petition; Effect of Non-Compliance with Requirements.
The petition shall contain the full names and actual addresses of all the petitioners and
respondents, a concise statement of the matters involved, the factual background of the case,
and the grounds relied upon for the relief prayed for.
x x x x
It shall be filedin seven (7) clearly legible copiestogether with proof of service thereof on the
respondentwith the original copy intended for the court indicated as such by the petitioner, and
shall be accompanied by a clearly legible duplicate original or certified true copy of the
judgment, order, resolution, or ruling subject thereof, such material portions of the record as are
referred to therein, and other documents relevant or pertinent thereto. The certification shall be
accomplished by the proper clerk of court or by his duly authorized representative, or by the
proper officer of the court, tribunal, agency or office involved or by his duly authorized
representative. The other requisite number of copies of the petition shall be accompanied by
clearly legible plain copies of all documents attached to the original.
The petitioner shall also submit together with the petition a sworn certification that he has not
theretofore commenced any other action involving the same issues in the Supreme Court, the
Court of Appeals or different divisions thereof, or any other tribunal or agency; if there is such
other action or proceeding, he must state the status of the same; and if he should thereafter
learn that a similar action or proceeding has been filed or is pending before the Supreme Court,
the Court of Appeals, or different divisions thereof, or any other tribunal or agency, he
undertakes to promptly inform the aforesaid courts and other tribunal or agency thereof within
five (5) days therefrom.
The petitioner shall pay the corresponding docket and other lawful fees to the clerk of court and
deposit the amount of P500.00 for costs at the time of the filing of the petition.
The failure of the petitioner to comply with any of the foregoing requirements shall be sufficient
ground for the dismissal of the petition.
SECTION 4.Jurisdiction Over Person of Respondent, How Acquired.
The court shall acquire jurisdiction over the person of the respondent by the service on him/[her]
of its order or resolution indicating its initial action on the petition or by his/[her] voluntary

submission to such jurisdiction.(Emphases and underscoring supplied)
Based on the above-quoted provisions, the petition must be accompanied by a proof of service
to respondents. Failure to comply with said requirement shall be a sufficient ground for the
dismissal of the petition. The Court shall acquire jurisdiction over the person of the respondent
upon service on him or her of its order or resolution indicating its initial action on the petition or
by voluntary submission.
As per the records, petitioner Republic served copies of the petition to the respondents through
Atty. Estelito Mendoza (Atty. Mendoza). However, respondents Lobregat and Concepcion turned
out to have not received their respective copies of the petition and all subsequent pleadings and
resolutions. Thus, they failed to file their Comment on the petition.
Patently, petitioner Republic's alleged failure to timely serve copies of the petition to
respondents Lobregat and Concepcion shall be sufficient ground for the dismissal of its Petition
under Rule 65. However, petitioner Republic clarified that such procedural infirmity was an
honest mistake as it relied on what was stated in the August 6, 1998 Review and
Recommendation and September 25, 1998 Order directing that copies thereof be sent to
respondents through Atty. Mendoza. Hence, petitioner Republic likewise served copies of the
petition to Lobregat and Concepcion through Atty. Mendoza.
Nonetheless, instead of dismissing the petition outright, this Court in its July 7, 2004 Resolution
consequently reversed and set aside the August 23, 2001 Decision and allowed the filing of the
Comments of all the respondents. It is worth noting that notice to adverse party is important to
prevent surprise and to afford the latter a chance to be heard in keeping with the principle of
procedural due process. However, it is also well-settled that procedural rules may be relaxed
when a "stringent application of [the same] would hinder rather than serve the demands of
substantial justice."
InSanchez v. Court of Appeals,We listed the elements to be considered to warrant the
suspension of the Rules, to wit:
Aside from matters of life, liberty, honor or property which would warrant the suspension of the
Rules of the most mandatory character and an examination and review by the appellate court of
the lower court's findings of fact, the other elements that should be considered are the following:
(a) the existence of special or compelling circumstances, (b) the merits of the case, (c) a cause
not entirely attributable to the fault or negligence of the party favored by the suspension of the
rules, (d) a lack of any showing that the review sought is merely frivolous and dilatory, and (e)
the other party will not be unjustly prejudiced thereby.
InGinete v. Court of Appeals,We explained the rationale in the relaxation of the rules of
procedure in case of justifiable instances, to wit:
Let it be emphasized that the rules of procedure should be viewed as mere tools designed to
facilitate the attainment of justice. Their strict and rigid application, which would result in
technicalities that tend to frustrate rather than promote substantial justice, must always be
eschewed. Even the Rules of Court reflect this principle. The power to suspend or even
disregard rules can be so pervasive and compelling as to alter even that which this Court itself
has already declared to be final, as we are now constrained to do in instant case.
Thus, this court is not averse to suspending its own rules in the pursuit of the ends of justice. "[x
x x] For when the operation of the Rules will lead to an injustice we have, in justifiable instances,
resorted to this extraordinary remedy to prevent it. The rules have been drafted with the primary
objective of enhancing fair trials and expediting justice. As acorollary,if their application and
operation tend to subvert and defeat, instead of promote and enhance it, their suspension is
justified. In the words of Justice Antonio P. Barredo in his concurring opinion in Estrada v. Sto.
Domingo, "(T)his Court, through the revered and eminent Mr. Justice Abad Santos, found
occasion in the case ofC. Viuda de Ordoveza v. Raymundo,to lay down for recognition in this
jurisdiction, the sound rule in the administration of justice holding that 'it is always in the power
of the court (Supreme Court) to suspend its own rules or to except a particular case from its
operation, whenever the purposes of justice require it [x x x]"
The Rules of Court were conceived and promulgated to set forth guidelines in the dispensation
of justice but not to bind and chain the hand that dispenses it, for otherwise, courts will be mere
slaves to or robots of technical rules, shorn of judicial discretion. That is precisely why courts, in
rendering justice have always been, as they in fact ought to be, conscientiously guided by the
norm that on the balance, technicalities take a backseat to substantive rights, and not the other

way around. As applied to instant case, in the language of Justice Makalintal, technicalities
"should give way to the realities of the situation."
Clearly, the present case pertains to the Ombudsman's investigation of respondents' purported
violation of RA 3019 allegedly involving government funds and/or property. The Republic should
not be faulted by the OSG's failure to timely serve copies of the petition to respondents
Concepcion and Lobregat within the reglementary period. Besides, petitioner Republic provided
justifiable reason for its failure to comply with the procedural requirements in filing the instant
petition. Also, to deny Republic's privilege to question the assailed OMB's August 6, 1998
Review and Recommendation and the September 25, 1998 Order would frustrate, rather than
promote, substantial justice, especially when the case involves purportedly public funds and/or
property. Hence, considering the existence of special or compelling circumstance, the technical
rules of procedure may be relaxed in this case in order to serve the demands of substantial
justice.
Grave abuse of discretion
Having resolved that the instant Petition under Rule 65 was correctly and timely filed in
accordance with the rules, We come now to the issue of grave abuse of discretion imputed
against the Ombudsman when it ordered the dismissal of OMB-0-90-2808 on the ground of
prescription.
Ordinarily, the Court does not interfere with the Ombudsman's determination as to the existence
or non-existence of probable cause except when there is grave abuse of discretion.As defined,
"grave abuse of discretion means such capricious or whimsical exercise of judgment which is
equivalent to lack of jurisdiction. To justify judicial intervention, the abuse of discretion must be
so patent and gross as to amount to an evasion of a positive duty or to a virtual refusal to
perform a duty enjoined by law or to act at all in contemplation of law, as where the power is
exercised in an arbitrary and despotic manner by reason of passion or hostility."

Notably, the Ombudsman's assailed Orders did not specifically rule on the existence or non-
existence of probable cause to indict or exonerate respondents Concepcion, Dela Cuesta,

Enrile, Ursua, and Pineda of violation of RA 3019. Instead, the Ombudsman ordered the
dismissal of the complaint on the ground of prescription.
After a careful review of the records, the Court finds judicial intervention is justified and proper in
this case to determine the correctness of the Ombudsman's order of dismissal on the ground of
prescription as per relevant laws and jurisprudence.
We emphasize that We are not ruling on the guilt or innocence of the respondents. Instead, Our
focus is on the plausible allegations of Republic, which may determine whether a violation of the
special law was apparent at the time of its commission.
Prescription of offense
To resolve the issues concerning prescription of offenses, the Court must determine the
following: (a) the prescriptive period of the offense; (b) when the period commenced to run; and
(c) when the period was interrupted.
(a) Prescriptive Period of the Offense
At the time of enactment of RA 3019, the original prescriptive period of offenses defined and
penalized therein was 10 years.Thereafter, on March 16, 1982, BP 195extended the prescriptive
period in filing cases for violation of RA 3019 from 10 years to 15 years. Subsequently, the
prescriptive period for violation of RA 3019 was extended to 20 years as per RA 10910,which
took effect on July 21, 2016.
It bears stressing that the Complaint charged respondents with violation of RA 3019 on account
of the execution of the MOA with AII on November 20, 1974. The prescriptive period during that
time for offenses punishable under RA 3019 was 10 years. Clearly, the amendatory laws,i.e.,BP
195 and RA 10910, which provide longer periods of prescription, cannot be retroactively applied
to crimes committed prior to their passage in 1982 and 2016, respectively.InPeople v.
Pacificador,the rule is that "in the interpretation of the law on prescription of crimes, that which is
more favorable to the accused is to be adopted."Therefore, the applicable prescriptive period in
the instant case is 10 years.
(b)When the period commenced to run
As to the reckoning point of the prescriptive period, RA 3019 fails to explicitly provide. Thus,
reference is to be made to Act No. 3326which governs the prescription of offenses punished by
special penal laws.

Sec. 2 of Act No. 3326 provides that prescription commences from: (a) the day of the
commission of the violation of the law, which is the general rule; or (b) if the same is not known,
from the time of discovery thereof and the institution of judicial proceeding for its investigation
and punishment, which is the exception and otherwise known as the discovery rule or the
blameless ignorance doctrine. The discovery rule or the blameless ignorance doctrine states:
SECTION 2.Prescription shall begin to run from the day of the commission of the violation of the
law, and if the same be not known at the time, from the discovery thereof and the institution of
judicial proceeding for investigation and punishment.
The prescription shall be interrupted when proceedings are instituted against the guilty person,
and shall begin to run again if the proceedings are dismissed tor reasons not constituting
jeopardy. (Emphasis supplied)
As elucidated inDel Rosario v. People,as a general rule, "the fact that any aggrieved person
entitled to an action has no knowledge of his/[her] right to sue or of the facts out of which
his/[her] right arises does not prevent the running of the prescriptive period."On the other hand,
the blameless ignorance rule provides that "the statute of limitations runs only upon discovery of
the fact of the invasion of a right which will support a cause of action."
InPresidential Commission on Good Government v. Carpio-Morales (Carpio-Morales),the Court
explains the construction of the discovery rule or the blameless ignorance doctrine and provides
guidelines in the determination of the reckoning point for the period of prescription of violations
of RA 3019, to wit:
The first mode being self-explanatory, We proceed with Our construction of the second mode.
In interpreting the meaning of the phrase "if the same be not known at the time, from the
discovery thereof and the institution of judicial proceeding for its investigation," this Court has,
as early as 1992 inPeople v. Duque,held that in cases where the illegality of the activity is not
known to the complainant at the time of its commission, Act No. 3326, Section 2 requires that
prescription, in such a case, would begin to run only from the discovery thereof, i.e., discovery of
the unlawful nature of the constitutive act or acts.
It is also inDuquewhere this Court espoused the raison d'être for the second mode. We said, "
[i]n the nature of things, acts made criminal by special laws are frequently not immoral or
obviously criminal in themselves; for this reason, the applicable statute requires that if the
violation of the special law is not known at the time, the prescription begins to run only from the
discovery thereof, i.e., discovery of the unlawful nature of the constitutive act or acts."
Further clarifying the meaning of the second mode, the Court, in Duque, held that Section 2
should be read as "[p]rescription shall begin to run from the day of the commission of the
violation of the law, and if the same be not known at the time, from the discovery thereof
anduntilthe institution of judicial proceedings for its investigation and punishment." Explaining
the reason therefor, this Court held that a contrary interpretation would create the absurd
situation where "the prescription period would both begin and be interrupted by the same
occurrence; the net effect would be that the prescription period would not have effectively
begun, having been rendered academic by the simultaneous interruption of that same period."
Additionally, this interpretation is consistent with the second paragraph of the same provision
which states that "prescription shall be interrupted when proceedings are instituted against the
guilty person, [and shall] begin to run again if the proceedings are dismissed for reasons not
constituting jeopardy."
Applying the same principle, We have consistently held in a number of cases, some of which
likewise involve behest loans contracted during the Marcos regime, that the prescriptive period
for the crimes therein involved generally commences from the discovery thereof, and not on the
date of its actual commission.
In the 1999 and 2011 cases ofPresidential Ad Hoc Fact-Finding Committee on Behest Loans v.
Desierto, the Court, in said separate instances, reversed the ruling of the Ombudsman that the
prescriptive period therein began to run at the time the behest loans were transacted and
instead, it should be counted from the date of the discovery thereof.
In the 1999 case, We recognized the impossibility for the State, the aggrieved party, to have
known the violation of RA 3019 at the time the questioned transactions were made in view of the
fact that the public officials concerned connived or conspired with the "beneficiaries of the
loans." There, We agreed with the contention of the Presidential Ad Hoc Fact-Finding
Committee that the prescriptive period should be computed from the discovery of the

commission thereof and not from the day of such commission. It was also in the same case
where We clarified that the phrase "if the same be not known" in Section 2 of Act No. 3326 does
not mean "lack of knowledge" but that the crime "is not reasonably knowable" is unacceptable.
Furthermore, in this 1999 case, We intimated that the determination of the date of the discovery
of the offense is a question of fact which necessitates the reception of evidence for its
determination.
Similarly, in the 2011 Desierto case, We ruled that the "blameless ignorance" doctrine applies
considering that the plaintiff therein had no reasonable means of knowing the existence of a
cause of action. In this particular instance, We pinned the running of the prescriptive period to
the completion by the Presidential Ad Hoc Fact-Finding Committee of an exhaustive
investigation on the loans. We elucidated that the first mode under Section 2 of Act No. 3326
would not apply since during the Marcos regime, no person would have dared to question the
legality of these transactions.
Prior to the 2011 Desierto case came Our 2006 Resolution inRomualdez v. Marcelo, which
involved a violation of Section 7 of RA 3019. In resolving the issue of whether or not the
offenses charged in the said cases have already prescribed, We applied the same principle
enunciated inDuqueand ruled that the prescriptive period for the offenses therein committed
began to run from the discovery thereof on the day former Solicitor General Francisco I. Chavez
filed the complaint with the PCGG.
This was reiterated inDisini v. Sandiganbayanwhere We counted the running of the prescriptive
period in said case from the date of discovery of the violation after the PCGG's exhaustive
investigation despite the highly publicized and well-known nature of the Philippine Nuclear
Power Plant Project therein involved, recognizing the fact that the discovery of the crime
necessitated the prior exhaustive investigation and completion thereof by the PCGG.
InRepublic v. Cojuangco, Jr.,however, We held that not all violations of RA 3019 require the
application of the second mode for computing the prescription of the offense. There, this Court
held that the second element for the second mode to apply, i.e., that the action could not have
been instituted during the prescriptive period because of martial law, is absent. This is so since
information about the questioned investment therein was not suppressed from the discerning
eye of the public nor has the Office of the Solicitor General made any allegation to that effect.
This Court likewise faulted therein petitioner for having remained dormant during the remainder
of the period of prescription despite knowing of the investment for a sufficiently long period of
time.
An evaluation of the foregoing jurisprudence on the matter reveals the following guidelines in the
determination of the reckoning point for the period of prescription of violations of RA 3019,viz.:
1. As a general rule, prescription begins to run from the date of the commission of the offense.
2. If the date of the commission of the violation is not known, it shall be counted form the date of
discovery thereof.
3. In determining whether it is the general rule or the exception that should apply in a particular
case, the availability or suppression of the information relative to the crime should first be
determined.
If the necessary information, data, or records based on which the crime could be discovered is
readily available to the public, the general rule applies. Prescription shall, therefore, run from the
date of the commission of the crime.
Otherwise, should martial law prevent the filing thereof or should information about the violation
be suppressed, possibly through connivance, then the exception applies and the period of
prescription shall be reckoned from the date of discovery thereof.(Emphasis supplied)
Applying the foregoing principles and based on Our judicious review of the records, We are
convinced that the exception on the date of discovery or the blameless ignorance doctrine
applies to the case at bar.
(i) The Republic could not have questioned the MOA because it was given legislative imprimatur.
It is worth noting that although the MOA dated November 20, 1974 was duly notarized and
presumably available to the public for scrutiny and perusal, the same was executed and entered
into by the parties pursuant to PD 582 issued by then President Marcos. Respondents even
contended, and the Ombudsman ruled in the assailed Orders, that the said MOA was given
legislative imprimatur. This allegedly implies that the respondents cannot be prosecuted for their
involvement in the execution, implementation, and termination of the said MOA. Hinging from

the same argument, the fact that the MOA dated November 20, 1974 was executed pursuant to
a legislative enactment, i.e., PD 582, the more it is highly impossible for the Republic to question
the same, and the respondents' alleged violation of RA 3019 and involvement in the execution,
implementation and termination of the MOA.
Hence, contrary to respondents' contention and the Ombudsman's assailed Orders, We are not
persuaded that the prescriptive period began to run in 1974 when the MOA with AII was
executed since petitioner Republic could not have possibly questioned the respondents for their
alleged violation of RA 3019 because it was given "legislative imprimatur" at that time. In other
words, it is not possible for the Republic, as the aggrieved party, to have known respondents'
alleged violation of RA 3019 prior to the 1986 Freedom Constitution which specifically mandated
the President to prioritize among others the: (a) recovery of ill-gotten properties amassed by the
leaders and supporters of the previous regime and protection of the interest of the people
through orders of sequestration or freezing of assets of accounts; and (b) eradication of graft
and corruption in government and punishment of those guilty thereof.Only then did the Republic
have the opportune time to discover acts or violations of RA 3019 in connection with the MOA
dated November 20, 1974 executed during the Marcos administration.
Similar toDisini v. Sandiganbayan,even arguing that the MOA dated November 20, 1974 is
publicly known as it involves government funds and affects the Philippine coconut industry, it
would have been futile for petitioner Republic to question the same and charge herein
respondents with violation of RA 3019 as no person would have dared to assail the legality of
MOA dated November 20, 1974 considering that President Marcos himself, exercising
legislative power, issued PD 582 which paved the way for the subject MOA.
Similar to PD 582, the amendments introduced in PD 961 and PD 1468 went unnoticed prior to
the date of discovery of the violation of RA 3019. To recall, both PD 961 and PD 1468 gave the
MOA an appearance of validity.
Sec. 3-B of PD 582 authorizes the execution of a contract for the financing of a hybrid coconut
seednut farm. Through PD 582, NIDC was given blanket authority to negotiate the contract on
behalf of the government.
With the amendments introduced by PD 961, a confirmatory phrase was added: "x x x the
contract entered into by NIDC as herein authorized is hereby confirmed and ratified; x x x."
While PD 582 paved the way for the MOA, PD 961 confirmed and ratified it.
Finally, upon further amendment by PD 1468, any amendment or supplement to the contract
was likewise confirmed and ratified. The phrase reads: "x x x the contract, including the
amendments and supplements thereto as provided for herein, entered into by NIDC as herein
authorized is hereby confirmed and ratified x x x." In effect, the series of supplemental
agreements and amendments subsequent to the MOA were confirmed and ratified.
With the legislative imprimatur of PD 582, PD 961, and PD 1468, it became nearly impossible
for petitioner Republic to question the MOA and its series of supplemental agreements and
amendments prior to the discovery of the offense. For this reason, the discovery rule or
blameless ignorance doctrine applies.
(ii) There were material subsequent events that transpired after the execution of the MOA, but
prior to the filing of the Complaint.
Apart from the disadvantageous provisions of the MOA, there are material subsequent events in
1982 and 1983 that transpired after the execution of the MOA. These material subsequent
events suggest the plausibility of a violation of RA 3019.
In the Complaint, petitioner Republic alleged that certain events transpired after the execution of
the MOA. These events include the following: (1) UCPB Board of Directors' adoption of
Resolution No. 111-83 on April 19, 1983;(2) UCPB Board of Directors' act of allowing the arbitral
award to lapse into finality;and (3) directorships of Enrile, Cojuangco, Jr., Dela Cuesta, and
Concepcion at AII until November 8, 1982,among many others. These materials events
transpired from 1982 to 1983, after the execution of the MOA and well before the filing of the
Complaint in 1990.
Appreciation of these events is necessary in determining when the prescriptive period
commenced to run because the acts of certain respondents corroborate their direct or indirect
participation in violation of RA 3019. We note that respondents Ursua and Pineda were neither
signatories to the MOA nor directors of AII. Nonetheless, respondents Ursua and Pineda were
members of UCPB's Board of Directors in 1983, whose acts still put the government at a

disadvantage. Thus, as to respondents Ursua and Pineda, the action is not barred by
prescription whether the general rule on date of commission or the exception on date of
discovery is applied.
Taken in its entirety, the material subsequent acts of respondents prove that any information
about the violation was suppressed. Thus, the discovery rule or blameless ignorance doctrine
applies.
(iii) The Complaint is replete with allegations of conspiracy and connivance.
InCarpio-Morales,We recognized that the reckoning point for the period of prescription of
violations of RA 3019 may commence on the date of discovery when information about the
violation of RA 3019 is suppressed, possibly through connivance.
Here, the Complaint is replete with allegations of conspiracy and connivance in the suppression
of information about the violation. Republic alleged as follows: (1) Cojuangco, Jr. took
advantage of his close relationship with then President Marcos for his own personal and
business interests through the issuance of favorable decrees;(2) Cojuangco, Jr. caused the
Philippine Government, through the NIDC, to enter into a contract with him, through AII, under
terms and conditions grossly disadvantageous to the government and in conspiracy with the

members of the UCPB Board of Directors, in flagrant breach of fiduciary duty as administrator-
trustee of the CIDF;(3) Enrile, Dela Cuesta, Concepcion, Ursua, and Pineda, as members of the

UCPB Board of Directors, allowed the BOA's March 29, 1983's Decision to lapse into finality,
which resulted in the successful siphoning of P840,789,855.33 from CIDF to AII;and (4)
respondents were directly or indirectly interested in personal gain, or had material interest in the
transaction requiring the approval of a board, panel, or group in which they were members, in
violation of RA 3019 to the grave damage and prejudice of the public interest, the Filipino
people, the Republic, and the coconut farmers.
In Our August 23, 2001 Decision, We deemed that the allegations of conspiracy and connivance
were sufficiently established in the pleadings, to wit:
There are striking parallelisms between the said Behest Loans Case and the present one which
lead us to apply the ruling of the former to the latter.First, both cases arouse out of seemingly
innocent business transactions;second,both were "discovered" only after the government
created bodies to investigate these anomalous transactions;third,both involve prosecutions for
violations of R.A. No. 3019; and,fourth,in both cases,it was sufficiently raised in the pleadings
that the respondents conspired and connived with one another in order to keep the alleged
violations hidden from public scrutiny.
x x x x
R.A. No. 3019, as applied to the instant case, covers not only the alleged one-sidedness of the
MOA, but also as to whether the contracts or transactions entered pursuant thereto by private
respondents were manifestly and grossly disadvantageous to the government, whether they
caused undue injury to the government, and whether the private respondents were interested
for personal gain or had material interests in the transactions.
The task to determine and find whether probable cause to charge private respondents exists
properly belongs to the Ombudsman. We only rule that the Office of the Ombudsman should not
have dismissed the complaint on the basis of prescription which is erroneous as hereinabove
discussed. The Ombudsman should have given the Solicitor General the opportunity to present
his evidence and then resolve the case for purposes of preliminary investigation. Failing to do
so, the Ombudsman acted with grave abuse of discretion.(Emphasis supplied)
Taken in its entirety and in view of the unique circumstance of this case, We declare that the
reckoning point of the prescriptive period should be from the promulgation of the 1986 Freedom
Constitution, which mandated the President to: (a) recover ill-gotten properties amassed by the
leaders and supporters of the previous regime and protect the interest of the people through
orders of sequestration or freezing of assets of accounts; and (b) eradicate graft and corruption
in government and punish those guilty thereof, among others. Only then will the Republic have
had the opportune time to discover any alleged acts or violations which would prompt the filing
of a necessary action against the culprits.
Therefore, petitioner Republic's Complaint dated February 12, 1990 filed against respondents
before the PCGG, which was subsequently referred to the Ombudsman, for violation of RA 3019
is well within the 10-year prescriptive period of an offense for the alleged illegal act committed
based on the MOA dated November 20, 1974.

(c) When the period was interrupted
Section 2 of Act No. 3326 clearly provides that prescription shall be interrupted when
proceedings are instituted against the accused, to wit:
SEC. 2. Prescription shall begin to ran from the day of the commission of the violation of the law,
and if the same be not known at the time, from the discovery thereof and the institution of
judicial proceedings for its investigation and punishment.
The prescription shall be interrupted when proceedings are instituted against the guilty person,
and shall begin to run again if the proceedings are dismissed for reasons not constituting
jeopardy.(Emphasis and underscoring supplied)
InPerez v. Sandiganbayan(Perez) citingPeople v. Pangilinan(Pangilinan), We declared
that"prescription is interrupted when the preliminary investigation against the accused is
commenced,"to wit:
Prescription is interrupted when the preliminary investigation against the accused is
commenced. InPeople v. Pangilinan, the Court held as follows:
x x x There is no more distinction between cases under the RPC and those covered by special
laws with respect to the interruption of the period of prescription. The ruling inZaldivia v. Reyes.
Jr.is not controlling in special laws.InLlenes v. Dicdican, Ingco, et al. v. Sandiganbayan, Brillante
v. CA,andSanrio Company Limited v. Lim,cases involving special laws, this Court held that the
institution of proceedings for preliminary investigation against the accused interrupts the period
of prescription. InSecurities and Exchange Commission v. Interport Resources Corporation, et
al.,the Court even ruled that investigations conducted by the Securities and Exchange
Commission for violations of the Revised Securities Act and the Securities Regulation Code
effectively interrupts the prescription period because it is equivalent to the preliminary
investigation conducted by the DOJ in criminal cases.
In fact, in the case of Panaguiton, Jr. v. Department of Justice, which is [on] all fours with the
instant case, this Court categorically ruled that commencement of the proceedings for the
prosecution of the accused before the Office of the City Prosecutor effectively interrupted the
prescriptive period for the offenses they had been charged under BP Big. 22. Aggrieved parties,
especially those who do not sleep on their rights and actively pursue their causes, should not be
allowed to suffer unnecessarily further simply because of circumstances beyond their control,
like the accused's delaying tactics or the delay and inefficiency of the investigating agencies.
(Emphasis in the original)
InPanaguiton, Jr. v. Department of Justice(Panaguiton), the Court explained the rationale for the
rule that prescription is interrupted by the commencement of the preliminary investigation, to wit:
It must be pointed out that when Act No. 3326 was passed on 4 December 1926, preliminary
investigation of criminal offenses was conducted by justices of the peace, thus, the phraseology
in the law, "institution of judicial proceedings for its investigation and punishment", and the
prevailing rule at the time was that once a complaint is filed with the justice of the peace for
preliminary investigation, the prescription of the offense is halted.
The historical perspective on the application of Act No. 3326 is illuminating. Act No. 3226 was
approved on 4 December 1926 at a time when the function of conducting the preliminary
investigation of criminal offenses was vested in the justices of the peace. Thus, the prevailing
rule at the time as shown in the cases ofU.S. v. LazadaandPeople v. Josonis that the
prescription of the offense is tolled once a complaint is filed with the justice of the peace for
preliminary investigation inasmuch as the filing of the complaint signifies the institution of the
criminal proceedings against the accused. These cases were followed by our declaration
inPeople v. ParaoandParaothat the first step taken in the investigation or examination of
offenses partakes the nature of a judicial proceeding which suspends the prescription of the
offense. Subsequently, inPeople v. Olarte, we held that the filing of the complaint in the
Municipal Court, even if it be merely for purposes of preliminary examination or investigation,
should, and does, interrupt the period of prescription of the criminal responsibility, even if the
court where the complaint or information is filed cannot try the case on the merits. In addition,
even if the court where the complaint or information is filed may only proceed to investigate the
case, its actuations already represent the initial step of the proceedings against the offender,
and hence, the prescriptive period should be interrupted.
InIngco v. Sandiganbayan and Sanrio Company Limited v. Lim,which involved violations of the
Anti-Graft and Corrupt Practices Act (R.A. No. 3019) and the Intellectual Property Code (R.A.

No. 8293), which are both special laws, the Court ruled that the prescriptive period is interrupted
by the institution of proceedings for preliminary investigation against the accused. In the more
recent case of Securities and Exchange Commission v. Inter port Resources Corporation, et al.,
the Court ruled that the nature and purpose of the investigation conducted by the Securities and
Exchange Commission on violations of the Revised Securities Act, another special law, is
equivalent to the preliminary investigation conducted by the DOJ in criminal cases, and thus
effectively interrupts the prescriptive period.
Panaguitonfurther held that to rule that the running of the prescriptive period is interrupted only
through the institution of judicial proceedings would deprive the injured party of his "right to
obtain vindication on account of delays that are not under his control."An aggrieved party who
actively pursues his or her cause should not be allowed to suffer unnecessarily simply because
of accused's delaying tactics or delay, and inefficiency of the investigating agencies.
Nonetheless, We are not unmindful of the rulings of this Court inJadewell Parking Systems
Corp. v. Judge Lidua, Sr.(Jadewell) andZaldivia v. Reyes, Jr.(Zaldivia) which declared that "the
running of the prescriptive period shall be halted on the date the case is actually filed in court
and not on any date before that"and "[a]s provided in the Revised Rules on Summary
Procedure, only the filing of an Information tolls the prescriptive period where the crime charged
is involved in an ordinance."
In other words, the Court ruled inJadewellandZaldiviathat when the offense involves violation of
a municipal or city ordinance, which is governed by the Revised Rules on Summary Procedure,
the running of the prescriptive period shall be interrupted only upon the institution of judicial
proceedings and not the commencement of the preliminary investigation by the investigating
agencies. In ruling so,JadewellandZaldiviamainly anchored on: (a) Sec. 9 of the 1983 Rules on
Summary Procedure, which substantially provides that the prosecution of criminal cases falling
under the summary procedure shall be either by complaint or by information filed directly in
court without need of a prior preliminary examination or preliminary investigation; and (b) Sec.
11 of the 1991 Revised Rules on Summary Procedure which provides that in Metropolitan
Manila and in Chartered Cities, the case is commenced only by Information except when the
offense cannot be prosecutedde oficio.
Patently,JadewellandZaldiviaare in apparent conflict withPanaguitonwhich involved a violation of
BP 22, which is also within the scope of the Revised Rules on Summary Procedure the sam
rules applicable on violation of municipal or city ordinance.
InPeople v. Lee, Jr.,the Court seemingly distinguished and reconciled the conflict
betweenJadewellandPanaguiton,which is affirmed inPeople v. Pangilinan,wherein the former
involved prescription for violation of ordinance while the latter refers to violation of special laws,
to wit:
The doctrine in the Panaguiton case was subsequently affirmed inPeople v. Pangilinan.In this
case, the affidavit-complaint forestafaand violation of B.P. Blg. 22 against the respondent was
filed before the Office of the City Prosecutor (OCP) of Quezon City on September 16, 1997. The
complaint stems from respondent's issuance of nine (9) checks in favor of private complainant
which were dishonored upon presentment and refusal of the former to heed the latter's notice of
dishonor which was made sometime in the latter part of 1995. On February 3, 2000, a complaint
for violation of BP Blg. 22 against the respondent was filed before the Metropolitan Trial
Court (MeTC) of Quezon City, after the Secretary of Justice reversed the recommendation of the
OCP of Quezon City approving the "Petition to Suspend Proceedings on the Ground of
Prejudicial Question" filed by the respondent on the basis of the pendency of a civil case for
accounting, recovery of commercial documents and specific performance which she earlier filed
before the Regional Trial Court of Valenzuela City. The issue of prescription reached this Court
after the Court of Appeals (CA), citing Section 2 of Act 3326, sustained respondent's position
that the complaint against her for violation of B.P. Blg. 22 had prescribed.
In reversing the CA's decision, We emphatically ruled that "(t)here is no more distinction
between cases under the RPC (Revised Penal Code) and those covered by special laws with
respect to the interruption of the period of prescription" and reiterated that the period of
prescription is interrupted by the filing of the complaint before the fiscal's office for purposes of
preliminary investigation against the accused.
In the case at bar, it was clear that the filing of the complaint against the respondent with the
Office of the Ombudsman on April 1, 2014 effectively tolled the running of the period of

prescription. Thus, the filing of the Information before the Sandiganbayan on March 21, 2017, for
unlawful acts allegedly committed on February 14, 2013 to March 20, 2014, is well within the
three (3)-year prescriptive period of R.A. No. 7877. The courta quo's reliance on the case
ofJadewell v. Judge Nelson Lidua, Sr.,is misplaced.Jadewellpresents a different factual milieu as
the issue involved therein was the prescriptive period for violation of a city ordinance, unlike
here as well as in thePangilinan and other above-mentioned related cases, where the issue
refers to prescription of actions pertaining to violation of a special law. For sure, Jadewell did not
abandon the doctrine inPangilinanas the former even acknowledged existing jurisprudence
which holds that the filing of complaint with the Office of the City Prosecutor tolls the running of
the prescriptive period.
It is worth noting that the offense inPanaguiton, i.e.,violation of BP 22, was committed in 1993
when BP 22 was not yet covered by the Revised Rules on Summary Procedure. In 2003, the
Supreme Court, through A.M. No. 00-11-01-SC,amended the Revised Rules on Summary
Procedure to include within its scope violations of BP 22. Thus, revisiting the rule on the
interruption of prescriptive period with respect to special laws and those offenses covered by
summary procedure is therefore in order.
Section 11 of the Revised Rules on Summary Procedure states that:
SECTION 11.How Commenced.The filing of criminal cases falling within the scope of thi
Rule shall be either by complaint or by information: Provided, however, that in Metropolitan
Manila and in Chartered Cities, such cases shall be commenced only by information, except
when the offense cannot be prosecutedde oficio.
The complaint or information shall be accompanied by the affidavits of the complainant and of
his witnesses in such number of copies as there are accused plus two (2) copies for the court's
files. If this requirement is not complied with within five (5) days from date of filing, the case may
be dismissed. (Emphasis supplied)
Patently, the phrase "without need of a prior preliminary examination or preliminary
investigation" found in Sec. 9 of the 1983 Rules on Summary Procedure is now deleted in the
above-quoted provision.Jadewelldeclared that "[a]s provided in the Revised Rules on Summary
Procedure, only the filing of an Information tolls the prescriptive period where the crime charged
is involved in an ordinance."Notably, the offense involved inJadewellis a violation of city
ordinance which, as provided in the Revised Rules on Summary Procedure, is commenced only
by information except when the offense cannot be prosecutedde oficio.
In other words, in Metropolitan Manila and in Chartered Cities, prescriptive period is tolled only
by the filing of an Information in court and not by the commencement of a preliminary
investigation by the investigating body nor the institution of the complaint with the investigating
body. Other than Metropolitan Manila and Chartered Cities, the criminal action is commenced by
filing a complaint or information before the court. In the same vein, the running of the
prescriptive period is interrupted by either the complaint or information filed in court.
Hence, for special laws within the scope of the Revised Rules on Summary Procedure, the
principle laid down inZaldiviaandJadewellis controlling, i.e. violations of municipal or city
ordinance, and BP 22. Accordingly, the ruling inPanaguitonwith respect to interruption of
prescription of BP 22 shall govern only those acts committed when BP 22 is not yet covered by
the Revised Rules on Summary Procedure,i.e.before the effectivity of A.M. No. 00-11-01-SC on
April 15, 2003. Thus, for acts committed on April 15, 2003 onwards, the filing of complaint or
information in court shall interrupt the running of the prescriptive period and not the institution of
the preliminary investigation by investigating agencies or the filing of a complaint before such
investigating agencies. However, in Metropolitan Manila and Chartered Cities, only the filing of
Information in court shall toll the running of the prescriptive period.
As to other special laws not covered by the Revised Rules on Summary Procedure, such as a
violation of RA 3019, the rule is that the prescriptive period is interrupted by the institution of
proceedings for preliminary investigation. Plainly, the ruling laid down inPerezandPangilinan,as
well as the justification elucidated inPanaguiton,are relevant and appropriate in the case at bar.
Hence, the filing of the instant complaint against respondents with the Office of the Ombudsman
in 1990 effectively tolled the running of the prescriptive period. From the reckoning
point,i.e.1986, only four years have lapsed when the Republic filed the Complaint in 1990
against respondents. Clearly, respondents' alleged violation of RA 3019 has not yet prescribed.

Moreover, the Complaint filed before the Ombudsman interrupted the running of the prescriptive
period. The respondents cannot, therefore, argue that the offense has already prescribed on the
basis of the absence of Information filed with the Sandiganbayan.
Ombudsman committed grave abuse of discretion when it dismissed the Complaint based on
prescription of offense
As a general rule, the Court cannot interfere with the Ombudsman's finding of probable cause
without violating the latter's constitutionally-granted investigatory and prosecutorial powers. Sec.
15 of RA 6770, otherwise known asThe Ombudsman Act,provides for the powers, functions and
duties of the Office of the Ombudsman, to wit:
SECTION 15.Powers, Functions and Duties. The Office of the Ombudsman shall have th
following powers, functions and duties:
(1) Investigate and prosecute on its own or on complaint by any person, any act or omission of
any public officer or employee, office or agency, when such act or omission appears to be
illegal, unjust, improper or inefficient. It has primary jurisdiction over cases cognizable by
theSandiganbayanand, in the exercise of this primary jurisdiction, it may take over, at any stage,
from any investigatory agency of Government, the investigation of such cases;
(2) Direct, upon complaint or at its own instance, any officer or employee of the Government, or
of any subdivision, agency or instrumentality thereof, as well as any government-owned or
controlled corporations with original charter, to perform and expedite any act or duty required by
law, or to stop, prevent, and correct any abuse or impropriety in the performance of duties;
(3) Direct the officer concerned to take appropriate action against a public officer or employee at
fault or who neglect to perform an act or discharge a duty required by law, and recommend his
removal, suspension, demotion, fine, censure, or prosecution, and ensure compliance therewith;
or enforce its disciplinary authority as provided in Section 21 of this Act:Provided,That the
refusal by any officer without just cause to comply with an order of the Ombudsman to remove,
suspend, demote, fine, censure, or prosecute an officer or employee who is at fault or who
neglects to perform an act or discharge a duty required by law shall be a ground for disciplinary
action against said officer;
(4) Direct the officer concerned, in any appropriate case, and subject to such limitations as it
may provide in its rules of procedure, to furnish it with copies of documents relating to contracts
or transactions entered into by his/[her] office involving the disbursement or use of public funds
or properties, and report any irregularity to the Commission on Audit for appropriate action;
(5) Request any government agency for assistance and information necessary in the discharge
of its responsibilities, and to examine, if necessary, pertinent records and documents;
(6) Publicize matters covered by its investigation of the matters mentioned in paragraphs (1),
(2), (3) and (4) hereof, when circumstances so warrant and with due prudence:Provided,That
the Ombudsman under its rules and regulations may determine what cases may not be made
public:Provided, further,That any publicity issued by the Ombudsman shall be balanced, fair and
true;
(7) Determine the causes of inefficiency, red tape, mismanagement, fraud, and corruption in the
Government, and make recommendations for their elimination and the observance of high
standards of ethics and efficiency;
(8) Administer oaths, issuesubpoenaandsubpoena duces tecum,and take testimony in any
investigation or inquiry, including the power to examine and have access to bank accounts and
records;
(9) Punish for contempt in accordance with the Rules of Court and under the same procedure
and with the same penalties provided therein;
(10) Delegate to the Deputies, or its investigators or representatives such authority or duty as
shall ensure the effective exercise or performance of the powers, functions, and duties herein or
hereinafter provided;
(11) Investigate and initiate the proper action for the recovery of ill-gotten and/or unexplained
wealth amassed after February 25, 1986 and the prosecution of the parties involved therein.
The Ombudsman shall give priority to complaints filed against high ranking government officials
and/or those occupying supervisory positions, complaints involving grave offenses as well as
complaints involving large sums of money and/or properties. (Emphasis supplied)
InPresidential Ad Hoc Committee on Behest Loans v. Tabasondra,the Court explained the
rationale behind the Court's non-interference with the Ombudsman's investigatory and

prosecutorial powers, to wit:
The Ombudsman has the power to investigate and prosecute any act or omission of a public
officer or employee when such act or omission appears to be illegal, unjust, improper or
inefficient.In fact, the Ombudsman has the power to dismiss a complaint without going through a
preliminary investigation, since he/[she] is the proper adjudicator of the question as to the
existence of a case warranting the filing of information in court. The Ombudsman has discretion
to determine whether a criminal case, given its facts and circumstances, should be filed or not.
This is basically his/[her] prerogative.
In recognition of this power, the Court has been consistent not to interfere with the
Ombudsman's exercise of his investigatory and prosecutory powers.
Various cases held that it is beyond the ambit of this Court to review the exercise of discretion of
the Office of the Ombudsman in prosecuting or dismissing a complaint filed before it. Such
initiative and independence are inherent in the Ombudsman who, beholden to no one, acts as
the champion of the people and preserver of the integrity of the public service.
The rationale underlying the Court's ruling has been explained in numerous cases.The rule is
based not only upon respect for the investigatory and prosecutory powers granted by the
Constitution to the Office of the Ombudsman but upon practicality as well. Otherwise, the
functions of the courts will be grievously hampered by innumerable petitions assailing the
dismissal of investigatory proceedings conducted by the Office of the Ombudsman with regard
to complaints filed before it, in much the same way that the courts would be extremely swamped
if they would he compelled to review the exercise of discretion on the part of the fiscals or
prosecuting attorneys each time they decide to file an information in court or dismiss a complaint
by a private complainant.In order to insulate the Office of the Ombudsman from outside
pressure and improper influence, the Constitution as well as Republic Act No. 6770 saw fit to
endow that office with a wide latitude of investigatory and prosecutory powers, virtually free from
legislative, executive or judicial intervention. If the Ombudsman, using professional judgment,
finds the case dismissible, the Court shall respect such findings unless they are tainted with
grave abuse of discretion.(Emphasis supplied)
It is worth noting that the instant petition is elevated before this Court via Rule 65 to determine
whether the Ombudsman committed grave abuse of discretion amounting to lack or in excess of
jurisdiction when it dismissed Republic's Complaint against respondents based on prescription
of offense. To reiterate, the Court generally does not interfere with the Office of Ombudsman in
its duty of finding the existence of probable cause nor its decision to dismiss the complaint
without undergoing preliminary investigation as in the case at bar which was dismissed by
reason of prescription of offense. An exception would be a finding of grave abuse of discretion.
As defined inCasing v. Ombudsman,"[g]rave abuse of discretion implies a capricious and
whimsical exercise of judgment tantamount to lack of jurisdiction. The Ombudsman's exercise of
power must have been done in an arbitrary or despotic manner which must be so patent an
gross as to amount to an evasion of a positive duty or a virtual refusal to perform the duty
enjoined or to act at all in contemplation of law in order to exceptionally warrant judicia
intervention."
As extensively discussed, Ombudsman Desierto's approval of the August 6, 1998 Review and
Recommendation and the September 25, 1998 Order which recommended the dismissal of the
Republic's Complaint based on prescription of offense is so patent and gross as to amount to an
evasion of a positive duty or virtual refusal to perform a duty enjoined, that is, to conduct a
preliminary investigation and to determine whether probable cause exists to charge herein
respondents with violation of RA 3019. As found by this Court, the dismissal based on
prescription of offense is erroneous and inconsistent with applicable law and jurisprudence.
Evidently, the Ombudsman should not have dismissed Republic's Complaint based on
prescription of offense, and proceeded to determine whether probable cause exists to charge
respondents with violation of RA 3019. The OSG should have been given an opportunity to
present evidence, and then resolve the case for purposes of preliminary investigation.
Nonetheless, it is premature for this Court to rule on the existence of probable cause and direct
the filing of the Information with the Sandiganbayan when the Ombudsman dismissed the
complaint not on the non-existence thereof, nor appreciation of the evidence, but on prescription
of offense. In other words, this Court cannot rule on whether there is probable cause to indict

respondents for violation of RA 3019, without interfering with the Ombudsman's investigatory
duty when the same was not even specifically considered as basis for the dismissal of the
Republic's Complaint.
In addition, this Court will not rule on respondent Concepcion's contention that he should not be
charged with violation of RA 3019 as he was merely impleaded in his capacity as a lawyer and
not in his own personal capacity. The issue calls for the discretionary power of the Ombudsman
to prosecute respondent Concepcion based on his involvement in the alleged anomaly
surrounding the MOA dated November 20, 1974.
Besides, the issue inRegalapertains to respondent Concepcion's alleged involvement as lawyer
and partner of ACCRA in relation to the Complaint dated July 31, 1987 filed by the Republic
against respondent Cojuangco, Jr. for the recovery of alleged ill-gotten wealth, which includes
shares of stocks in the named corporations in SB Civil Case No. 0033 entitledRepublic of the
Philippines v. Eduardo Cojuangco.
SB Civil Case No. 0033 alleged that respondents Concepcion and Cojuangco, Jr. and other
defendants therein conspired in setting up, through the use of coconut levy funds, the financial
and corporate framework and structures that led to the establishment of UCPB, UNICOM, and
through insidious means and machinations, ACCRA, using its wholly-owned investment arm,
ACCRA Investments Corporation, became the holder of approximately 15 million shares
representing roughly 3.3% of the total capital stock of UCPB as of March 31, 1987.In
fine,Regalaexcluded respondent Concepcion and other ACCRA lawyers from SB Civil Case No.

0033 based on the privilege of attorney-client confidentiality, constitutional right against self-
incrimination, and equal protection clause.

On the other hand, the present criminal case concerns respondent Concepcion's alleged
involvement in the MOA dated November 20, 1974 which purportedly violated RA 3019. To
reiterate, the duty to prosecute respondent Concepcion is within the discretionary power of
Ombudsman based on its own finding of probable cause.
Speedy Disposition
Finally, respondents allege that the delay in the filing of the necessary Information before the
Sandiganbayan violated their constitutional right to speedy disposition of cases.
The right to speedy disposition of cases is embodied under Sec. 16, Art. III of the
Constitution,viz.:
Section 16. All persons shall have the right to a speedy disposition of their cases before all
judicial, quasi-judicial, or administrative bodies.
Furthermore, Sec. 12, Art. XI of the Constitution requires the Ombudsman to act promptly on all
complaints filed before it:
Section 12. The Ombudsman and his[/her] Deputies, as protectors of the people, shall act
promptly on complaints filed in any form or manner against public officials or employees of the
Government, or any subdivision, agency or instrumentality thereof, including government-owned
or controlled corporations, and shall, in appropriate cases, notify the complainants of the action
taken and the result thereof.
Also, Sec. 13 of RA 6770 mandates the Ombudsman to:
Section 13.Mandate. The Ombudsman and his[/her] Deputies, as protectors of the people
shall act promptly on complaints filed in any form or manner against officers or employees of the

government, or of any subdivision, agency or instrumentality thereof, including government-
owned or controlled corporations, and enforce their administrative, civil and criminal liability in

every case where the evidence warrants in order to promote efficient service by the Government
to the people.
InCagang v. Sandiganbayan(Cagang), there was inordinate delay by the Sandiganbayan in the
resolution and termination of preliminary investigation. The Court laid down the guidelines to
resolve issues involving the right to speedy disposition of cases, to wit:
First,the right to speedy disposition of cases is different from the right to speedy trial. While the
rationale for both rights is the same, the right to speedy trial may only be invoked in criminal
prosecutions against courts of law.The right to speedy disposition of cases, however, may be
invoked before any tribunal, whether judicial or quasi-judicial.What is important is that the
accused may already be prejudiced by the proceeding for the right to speedy disposition of
cases to be invoked.

Second,a case is deemed initiated upon the filing of a formal complaint prior to a conduct of a
preliminary investigation.This Court acknowledges, however, that the Ombudsman should set
reasonable periods for preliminary investigation, with due regard to the complexities and
nuances of each case. Delays beyond this period will be taken against the prosecution. The
period taken for fact-finding investigations prior to the filing of the formal complaint shall not be
included in the determination of whether there has been inordinate delay.
Third,courts must first determine which party carries the burden of proof.If the right is invoked
within the given time periods contained in current Supreme Court resolutions and circulars, and
the time periods that will be promulgated by the Office of the Ombudsman, the defense has the
burden of proving that the right was justifiably invoked. If the delay occurs beyond the given time
period and the right is invoked, the prosecution has the burden of justifying the delay.
If the defense has the burden of proof, it must prove first, whether the case is motivated by
malice or clearly only politically motivated and is attended by utter lack of evidence, and second,
that the defense did not contribute to the delay.
Once the burden of proof shifts to the prosecution, the prosecution must prove first, that it
followed the prescribed procedure in the conduct of preliminary investigation and in the
prosecution of the case; second, that the complexity of the issues and the volume of evidence
made the delay inevitable; and third, that no prejudice was suffered by the accused as a result
of the delay.
Fourth,determination of the length of delay is never mechanical. Courts must consider the entire
context of the case, from the amount of evidence to be weighed to the simplicity or complexity of
the issues raised.
An exception to this rule is if there is an allegation that the prosecution of the case was solely
motivated by malice, such as when the case is politically motivated or when there is continued
prosecution despite utter lack of evidence. Malicious intent may be gauged from the behavior of
the prosecution throughout the proceedings. If malicious prosecution is properly alleged and
substantially proven, the case would automatically be dismissed without need of further analysis
of the delay.
Another exception would be the waiver of the accused to the right to speedy disposition of cases
or the right to speedy trial. If it can be proven that the accused acquiesced to the delay, the
constitutional right can no longer be invoked.
In all cases of dismissals due to inordinate delay, the causes of the delays must be properly laid
out and discussed by the relevant court.
Fifth,the right to speedy disposition of cases or the right to speedy trial must be timely
raised.The respondent or the accused must file the appropriate motion upon the lapse of the
statutory or procedural periods. Otherwise, they are deemed to have waived their right to
speedy disposition of cases.(Emphasis supplied)
We applyCagangto the case at bar. The Court finds that respondents Concepcion, Dela Cuesta,
Enrile, Ursua, and Pineda's constitutional right to speedy disposition of cases was violated by
the Ombudsman through the inordinate delay in concluding the preliminary investigation.
Below is a timeline of incidents from the filing of the Complaint:
Based on this timeline, it is apparent that the preliminary investigation spanned for over eight
years. It was only in 1997 that any movement or action on the case actually began.
Cagangemphasizes that it is important to determine who has the burden of proving delay, If the
delay is beyond the time periods provided in the rules, then the burden shifts to the State, or in
this case, to petitioner Republic.
Here, respondents argue that their right to speedy disposition of cases was violated by the
Ombudsman. To determine whether the delay is inordinate,Caganginstructs the Court to
examine whether the Ombudsman followed the specified time periods for the conduct of
preliminary investigation.FollowingCagang,the subsequent rulings inJavier v.
Sandiganbayan(Javier) andCatamco v. Sandiganbayan(Catamco) emphasized that the
Ombudsman rules did not specify time periods to conclude preliminary investigations.Thus, the
Court deemed the time periods provided in the Rules of Court to have suppletory application to
proceedings before the Ombudsman.
The recent case ofLorenzo v. Hon. Sandiganbayan Sixth Division(Lorenzo) involves prosecution
for violation of RA 3019. The case ofLorenzostemmed from the alleged anomalous procurement
of various quantities of fertilizer (granular urea) from the Philippine Phosphate Fertilizer

Corporation for the Luzon regions in 2003 by government officials of the Department of
Agriculture and National Food Authority.
InLorenzo,the Court elucidated on the right of speedy disposition of cases by applyingCagang,
Javier,andCatamco.Thereafter, this Court found that there was a violation of the constitutional
right to speedy disposition of cases when the preliminary investigation spanned four years from
the filing of the complaint to the approval of an Order denying a motion for reconsideration.
We quote below the discussion inLorenzoand the applicable time periods for fact-finding
investigations:
In the absence of specific time periods in the Rules of the Ombudsman,JavierandCatamcothus
applied Section 3, Rule 112 of the Revised Rules of Criminal Procedure, which provides that the
investigating prosecutor has 10 days after the investigation to determine whether there is
sufficient ground to hold the respondent for trial. This 10-day period may appear short or
unreasonable from an administrative standpoint. However, as held inAlarilla v.
Sandiganbayan (Alarilla), given the Court's duty to balance the right of the State to prosecute
violations of its lawvis-à-visthe rights of citizens to speedy disposition of cases, the citizens
ought not to be prejudiced by the Ombudsman's failure to provide for particular time periods in
its own Rules of Procedure.
Soon after the promulgation ofJavierandCatamco,the Ombudsman issued Administrative Order
No. (A.O.) 1 series of 2020 which specified the time periods in conducting its investigations.
For fact-finding Investigations, A.O. 1 provides that "[u]nless otherwise provided for in a
separate issuance, such as an Office Order creating a special panel of investigators and
prescribing therein the period for the completion of an investigation, the period for completion of
the investigation shall not exceed six (6) months for simple cases and twelve (12) months for
complex cases" subject to considerations on the complexity of the case and the possibility of
requesting for extension on justifiable reasons, which shall not exceed one year. Notably, the
factfinding investigation in this case arguably spanned 10 years, or from October 2003 until
November 2013 when the Complaint was filed before the Ombudsman, which is clearly beyond
the period provided in A.O. 1. Nevertheless, the Court is constrained to disregard this apparent
delay following the prevailing doctrine inCagangthat the period taken for fact-finding
investigations prior to the filing of the formal complaint shall not be included in the determination
of whether there has been inordinate delay.
We reproduce the relevant portions of Administrative Order No. (A.O.) 1, series of 2020on the
applicable time periods:
Section 7.Commencement of Preliminary Investigation. -Without prejudice to the Procedure in
Criminal Cases prescribed under Rule II of Administrative Order No. 07, as amended,a
preliminary investigation is deemed to commence whenever a verified complaint, grievance or
request for assistance is assigned a case docket numberunder any of the following instances:
a) Upon referral by an Ombudsman case evaluator to the preliminary investigation units/offices
of the Office of the Ombudsman, after determining that the verified complaint, grievance or
request for assistance is sufficient in form and substance and establishes the existence of
aprima faciecase against the respondent/s; or
b) At any time before the lapse; of the period for the conduct of a fact-finding investigation
whenever the results thereof support a finding ofprima faciecase.
In all instances, the complaint, grievance or request for assistance with an assigned case docket
number shall be considered as pending for purposes of issuing an Ombudsman clearance.
Section 8.Period for the conduct of Preliminary Investigation.- Unless otherwise provided for in a
separate issuance, such as an Office Order creating a special panel of investigators/prosecutors
and prescribing the period for completion of the preliminary investigation, the
proceedingstherein shall not exceed twelve (12) months for simple cases or twenty-four months
(24) months for complex cases,subject to the following considerations:
a) The complexity of the case shall be determined on the basis of factors such as, but not limited
to, the number of respondents, the number of offenses charged, the volume of documents, the
geographical coverage, and the amount of public funds involved.
b) Any delay incurred in the proceedings, whenever attributable to the respondent, shall
suspend the running of the period for purposes of completing the preliminary investigation.
c) The period herein prescribed may be extended by written authority of the Ombudsman, or the
Overall Deputy Ombudsman/Special Prosecutor/Deputy Ombudsman concerned for justifiable

reasons, which extension shall not exceed one (1) year.
Section 9.Termination of Preliminary Investigation. A preliminary investigation shall b
deemed terminatedwhen the resolution of the complaint, including any motion for
reconsideration filed in relation to the result thereof,as recommended by the Ombudsman
investigator/prosecutor and their immediate supervisors,is approved by the Ombudsmanor the
Overall Deputy Ombudsman/Special Prosecutor/Deputy Ombudsman concerned. (Emphasis
supplied)
Applying the foregoing to the instant case, preliminary investigation commenced on February
12, 1990 when the Complaint was filed, and terminated on October 9, 1998 when the
Ombudsman approved the Order dated September 25, 1998 and denied the Republic's motion
for reconsideration. Thus, whether the Court applies the 10-day period in Javier and Catamco,
or the more generous periods of 12 to 24 months under A.O. 1, We arrive at the same
conclusion that the Ombudsman exceeded the specified period provided for preliminary
investigations.
Consequently, the burden of proof shifted to petitioner Republic. However, petitioner Republic
failed to discharge this burden, as petitioner Republic did not establish that the delay was
reasonable and justified. In particular, petitioner Republic did not prove that: (1) it followed the
prescribed procedure in the conduct of preliminary investigation and the prosecution of the case;
(2) the complexity of the issues and the volume of evidence made the delay inevitable; and (3)
no prejudice was suffered by the accused as a result of the delay.
Cagangstates that Courts must consider the entire context of the case, from the amount of
evidence to be weighed to the simplicity or complexity of the issues raised. The Court observes
that there is no elucidation in petitioner Republic's pleadings as to what specific issue is too
complex or what voluminous records are involved to justify the delay. To be sure, matters not
involving complex factual or legal issues should not take long to resolve.1âшphi1
By way of exception, if the accused acquiesced to the delay, then the constitutional right to
speedy disposition of cases cannot be invoked. As held inPeople v.
Sandiganbayan,citingCagang,the invocation of the right to speedy disposition of a case must be
timely raised through an appropriate motion; otherwise, the delay would be construed as
acquiesced or waived.
It is worth noting that not one of the respondents invoked their right to speedy disposition of
cases before the Ombudsman during the preliminary investigation stage prior to the issuance of
the assailed August 6, 1998 Review and Recommendation and the September 25, 1998 Order
as approved by Ombudsman Desierto on August 14, 1998 and October 9, 1998, respectively.
However, as respondents, they had no duty to expedite or follow-up the cases against them
since there are determined periods for the termination of the preliminary investigation.Thus, the
mere inaction on the part of accused, without more, does not qualify as an intelligent waiver of
their constitutionally guaranteed right to the speedy disposition of cases.
In fact, the earliest opportunity for respondents to invoke their constitutional right to speedy
disposition of cases was before this Court in response to the present Petition by the Republic.
Among the respondents, Dela Cuesta argued that "this is the only opportune time for
respondent to invoke his right'"because he was not served a copy of the Petition at the outset.
Nonetheless, respondents' failure to invoke their constitutional right is not fatal to their cause.
Additionally, the Republic failed to show that petitioners did not suffer any prejudice because of
the 8-year delay.Cagang,citingCorpuz v. Sandiganbayan,explains the concept of prejudice, to
wit:
Prejudice should be assessed in the light of the interest of the defendant that the speedy trial
was designed to protect, namely: to prevent oppressive pretrial incarceration; to minimize
anxiety and concerns of the accused to trial; and to limit the possibility that his [or her] defense
will be impaired. Of these, the most serious is the last, because the inability of a defendant
adequately to prepare his case skews the fairness of the entire system.There is also prejudice if
the defense witnesses are unable to recall accurately the events of the distant past. Even if the
accused is not imprisoned prior to trial, he is still disadvantaged by restraints on his liberty and
by living under a cloud of anxiety, suspicion and often, hostility. His financial resources may be
drained, his association is curtailed, and he is subjected to public obloquy.(Emphasis supplied)
With this case pending for over 30 years and possibly more without assurance of its resolution,
the Court recognizes that the tactical disadvantages carried by the passage of time should be

weighed against petitioner Republic and in favor of the respondents.Certainly, if this case were
remanded for further proceedings, the already long delay would drag on longer. Memories fade,
documents and other exhibits can be lost and vulnerability of those who are tasked to decide
increase with the passing of years.In effect, there would be a general inability to mount an
effective defense.
Taken in its entirety, there is a clear violation of the respondents' constitutional right to speedy
disposition of cases when petitioner Republic failed to provide sufficient justification for the delay
in the termination of the preliminary investigation. Consequently, a dismissal of the case is
warranted.
While this Court has no doubt that the Republic had all the resources to pursue cases of
corruption and ill-gotten wealth, the inordinate delay in this case may have made the situation
worse for respondents.
As the Supreme Court, We dutifully exercise cold impartiality while demanding accountability
from the government and protecting the rights of all people.
WHEREFORE, the petition isPARTIALLY GRANTED. Accordingly:
1. The August 6, 1998 Review and Recommendation and the September 25, 1998 Order in
OMB-0-90-2808, as approved by Ombudsman Aniano A. Desierto on August 14, 1998 and
October 9, 1998, respectively, areREVERSEDandSET ASIDE;
2. Due to their supervening deaths, the Complaint for violation of Republic Act No. 3019
docketed as OMB-0-90-2808 isDISMISSEDand the case isCLOSEDandTERMINATEDas
against respondents Eduardo M. Cojuangco, Jr., Jose R. Eleazar, Jr., Maria Clara Lobregat, and
Augusto Orosa. Consequently, their criminal liabilities and civil liability ex delicto are
extinguished by Article 89 of the Revised Penal Code. However, for civil liability based on
sources other than delict, petitioner Republic of the Philippines may file a separate civil action
against the respective estates of Eduardo M. Cojuangco, Jr., Jose R. Eleazar, Jr., Maria Clara
Lobregat, and Augusto Orosa as may be warranted by law and procedural rules; or if already
filed, the said separate civil action shall survive notwithstanding the dismissal of the criminal
case in view of their deaths; and
3. Due to the violation of the constitutional right to speedy disposition of cases, the Ombudsman
is hereby ordered toDISMISSthe Complaint for violation of Republic Act No. 3019 docketed as
OMB-0-90-2808 against respondents Jose C. Concepcion, Rolando Dela Cuesta, Juan Ponce
Enrile, Narciso M. Pineda, and Danilo S. Ursua.
SO ORDERED.
Zalameda, M. Lopez, Rosario, and Marquez, JJ., concur.
"""
processed_example = preprocess_text(example)
chunks = split_text(processed_example, tokenizer, overlap=100)
all_results = []
for chunk in chunks:
    ner_results = nlp(chunk)
    all_results.extend(ner_results)
chunked_results = manual_chunking(all_results)
filtered_results = filter_entities(chunked_results)

# Print final results
for chunk in filtered_results:
    print(f"Entity: {chunk['word']}, Label: {chunk['label']}, Probability: {chunk['score']:.2f}%, Start: {chunk['start']}, End: {chunk['end']}\n")
