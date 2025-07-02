import torch
from tqdm import tqdm
from services.huggingface.contrastive.model_loader import PredictionHead
from sklearn.metrics.pairwise import cosine_similarity

model : PredictionHead = PredictionHead(input_dim=2048, output_dim=2048)
model.load_state_dict(torch.load("output/v6/model_epoch_3_step_269.pth", map_location=torch.device('cpu')))

anchor : str = """# Rell\n\n## Support\n\n### Classes\nVanguard, Catcher\n\n### Key Abilities\n- Ferromancy (Mount/Dismount Mechanic)\n- Magnet Storm\n- Shattering Strike\n\n### Playstyle Summary\nRell is an aggressive engage support with powerful crowd control and defensive capabilities. Rell's unique **Ferromancy** mount/dismount mechanic allows her to switch between high-mobility engagement and defensive positioning, giving Rell unparalleled versatility in team compositions. She excels at initiating fights by pulling enemies together with **Magnet Storm**, disrupting enemy formations with knockups and stuns, and protecting her allies with shields and resistance-shredding abilities. Rell's kit is designed to create chaos in teamfights, disable key targets, and provide substantial defensive buffs to herself and her allies. What makes Rell particularly effective is her ability to control the battlefield through crowd control and her ability to reduce enemy defenses while simultaneously boosting her own survivability.\n"""
positive : str = """# Leona\n\n## Support\n\n### Classes\nVanguard, Catcher\n\n### Key Abilities\n- Sunlight (Passive): Synergistic mark that allows allies to deal bonus magic damage\n- Zenith Blade: Engage tool with crowd control and mobility\n- Solar Flare: High-impact ultimate with strong crowd control potential\n\n### Playstyle Summary\nLeona is a tanky engage support specialized in initiating fights and controlling the battlefield through crowd control. Leona's kit revolves around stunning and rooting enemies while providing defensive capabilities through her **Eclipse** ability. She excels at setting up kills for her allies by marking targets with **Sunlight** and using her crowd control abilities like **Shield of Daybreak** and **Solar Flare** to lock down opponents. What makes Leona particularly effective is her high defensive stats and damage reduction from **Eclipse**, allowing Leona to survive extended engagements and maintain her frontline presence. This durability makes her a formidable frontline support who can create opportunities for her team to follow up on Leona's aggressive plays, as she can reliably engage and disrupt enemy positioning while her allies capitalize on the openings she creates.\n"""
negative_1 : str = """# Lulu\n\n## Support\n\n### Classes\nEnchanter, Catcher\n\n### Key Abilities\n- **Pix, Faerie Companion**: Unique passive that allows Pix to assist in damage and attacks\n- **Whimsy**: Versatile ability that can polymorph enemies or boost ally stats\n- **Wild Growth**: Powerful defensive ultimate with crowd control and defensive enhancement\n\n### Playstyle Summary\nLulu is a versatile support champion who excels at protecting and empowering allies while disrupting enemies. Lulu's kit revolves around **Pix**, her faerie companion, who provides additional damage and utility. She can polymorph enemies to neutralize their threat, shield and speed up allies, and use her ultimate to provide massive defensive buffs and crowd control. What makes Lulu particularly effective is her ability to turn teamfights through strategic use of crowd control and ally enhancement, making her excellent at peeling for carries and creating playmaking opportunities. Lulu's success in team compositions stems from her unique combination of offensive disruption and defensive utility. However, when playing Lulu, players must rely heavily on precise timing and positioning to maximize her impact, as Lulu's abilities require careful coordination to achieve their full potential.\n"""
negative_2 : str = """# Lux\n\n## Midlane\n\n### Classes\nBurst Mage, Enchanter\n\n### Key Abilities\n- **Illumination** (Passive mark and bonus damage)\n- **Light Binding** (Root and crowd control)\n- **Prismatic Barrier** (Defensive shield for allies)\n- **Lucent Singularity** (Area control and damage)\n- **Final Spark** (Long-range burst damage)\n\n### Playstyle Summary\nLux is a versatile burst mage who excels at controlling team fights from a distance while providing utility to her allies. Lux's kit focuses on long-range crowd control and damage, with the ability to root enemies, slow their movement, and deal significant magic damage through her abilities. The **Illumination** passive encourages active play by rewarding Lux players who weave basic attacks between spell casts. Lux's **Prismatic Barrier** provides defensive support to teammates, while her **Final Spark** allows Lux to execute low-health targets or deal massive damage from a safe distance. As a champion, Lux is most effective when positioning carefully, using her long-range abilities to control the battlefield and burst down isolated targets. When playing Lux optimally, players can dominate team fights through superior positioning and precise ability timing.\n"""

anchor_embedding : torch.Tensor = model(anchor).detach().cpu().to(torch.float32)
positive_embedding : torch.Tensor = model(positive).detach().cpu().to(torch.float32)
negative_embedding_1 : torch.Tensor = model(negative_1).detach().cpu().to(torch.float32)
negative_embedding_2 : torch.Tensor = model(negative_2).detach().cpu().to(torch.float32)

sim1 = cosine_similarity(anchor_embedding, positive_embedding)
sim2 = cosine_similarity(anchor_embedding, negative_embedding_1)
sim3 = cosine_similarity(anchor_embedding, negative_embedding_2)
print(f"Similarity between anchor and positive: {sim1[0][0]}")
print(f"Similarity between anchor and negative_1: {sim2[0][0]}")
print(f"Similarity between anchor and negative_2: {sim3[0][0]}")

# good_predictions : int = 0
# bad_predicitons : int = 0
# total_predictions : int = 0

# for example in tqdm(dataset):
#     e_anchor = model(example["anchor"]).detach().cpu().to(torch.float32)
#     e_positive = model(example["positive"]).detach().cpu().to(torch.float32)
#     e_negative = model(example["negative"]).detach().cpu().to(torch.float32)
#     sim1 = cosine_similarity(e_anchor, e_positive)[0][0]
#     sim2 = cosine_similarity(e_anchor, e_negative)[0][0]
    
#     if sim1 > sim2:
#         good_predictions += 1
#     elif sim1 < sim2:
#         bad_predicitons += 1
#     total_predictions += 1
    
# print(f"Good predictions: {good_predictions} ({good_predictions / total_predictions * 100:.2f}%)")
# print(f"Bad predictions: {bad_predicitons} ({bad_predicitons / total_predictions * 100:.2f}%)")
# print(f"Total predictions: {total_predictions}")

# print(anchor_embedding.numpy().tolist())
# print(positive_embedding)
# print(negative_embedding)


