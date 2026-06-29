# ShipRSImageNet

## Overview

**ShipRSImageNet** is a large-scale, fine-grained dataset for **ship detection in
high-resolution optical remote sensing (satellite/aerial) imagery**. It targets
the *fine-grained* problem: not just "is there a ship?", but *which specific type
of ship* — distinguishing, for example, a Nimitz-class carrier from a
Ticonderoga-class cruiser, or a container ship from a RoRo.

- **Domain:** top-down optical remote sensing images of ports, harbours, and open
  sea, collected from multiple satellite/sensor platforms across worldwide
  locations, seasons, and image qualities.
- **Scale:** 3,435 images, each ~930×930 pixels, with **17,573 annotated ship
  instances**. Ships vary widely in scale, orientation, and aspect ratio.
- **Annotations:** every instance is labelled with a **horizontal bounding box
  (HBB)**, an **oriented bounding box (OBB)**, and a **polygon/mask** — providing
  direction, background, sea environment, and location information.
- **Sources:** xView (WorldView-3, 0.3 m GSD), HRSC2016, FGSD, the Airbus Ship
  Detection Challenge, and Chinese satellites (GaoFen-2, JiLin-1). Some imagery
  derives from Google Earth.
- **License:** academic use only — commercial use is prohibited.
- **Versioning:** V1.1 adds a dedicated, standardized **test set** for fair
  benchmarking. Benchmarks were produced with MMDetection v2.11.0.

## How the classes are grouped (4-level hierarchy)

Labels are organised as a **classification tree** with four levels, so a model
can be trained at whatever granularity a task needs:

| Level | Name | What it captures |
|-------|------|------------------|
| **Level 0** | *Class* | Whether the object is a ship at all (ship vs. non-ship). |
| **Level 1** | *Category* | Broad category of the ship (e.g. warship, merchant, auxiliary). |
| **Level 2** | *Sub-category* | Finer split within a Level-1 category. |
| **Level 3** | *Type* | The specific ship type/class — **50 categories total**. |

Broadly, the 50 Level-3 types fall into these families:

- **Warships / combatants:** aircraft carriers, destroyers (DD), frigates (FF),
  cruisers, submarines, patrol craft.
- **Amphibious / landing ships (LL):** landing ships, dock landing ships (LSD),
  and landing helicopter assault ships (LHA).
- **Auxiliary / support ships (AS):** replenishment ships (AOE), medical, test,
  training, command, and fast transport (EPF) vessels.
- **Merchant / civilian:** container ships, RoRo, cargo, tankers, ferries,
  fishing vessels, tugs, barges, plus small craft (yachts, sailboats, motorboats).
- **Fallback "Other" classes** at each branch absorb instances that don't match a
  named ship class.

### Abbreviations used in class names

| Abbr. | Meaning |
|-------|---------|
| DD | Destroyer |
| FF | Frigate |
| LL | Landing (ship) |
| AS | Auxiliary Ship |
| LSD | Landing Ship Dock |
| LHA | Landing Helicopter Assault ship |
| AOE | Fast Combat Support Ship |
| EPF | Expeditionary Fast Transport ship |
| RoRo | Roll-on / Roll-off ship |

## The 50 ship classes

> Class names follow the dataset's Level-3 naming exactly. Specific named classes
> (e.g. *Arleigh Burke DD*) refer to a real navy ship class whose distinctive
> top-down silhouette the detector learns to recognise.

| # | Class | Description |
|---|-------|-------------|
| 1 | **Other Ship** | Catch-all fallback for any ship that doesn't fit a more specific category. Used when the silhouette is a ship but its type can't be confidently assigned. Sits at the bottom of the hierarchy as a generic "ship" bucket. |
| 2 | **Other Warship** | Generic military combatant that doesn't match one of the named warship classes. Covers naval vessels with weapon/superstructure cues but no recognisable specific class. Acts as the fallback within the warship branch. |
| 3 | **Submarine** | Naval underwater vessel, appearing as a long, narrow, low-profile dark hull, often alongside a pier. Lacks the tall superstructure of surface combatants. Distinct cylindrical shape aids recognition from above. |
| 4 | **Other Aircraft Carrier** | Aircraft carrier not matching a named carrier class. Identified by a very large flat flight deck. Fallback within the carrier sub-category. |
| 5 | **Enterprise** | USS *Enterprise* (CVN-65), the first nuclear-powered aircraft carrier. A single, distinctive supercarrier with a long flat deck and island superstructure. One of the largest warship silhouettes in the dataset. |
| 6 | **Nimitz** | Nimitz-class nuclear-powered supercarrier (US Navy). Very large flat-deck carriers with an angled flight deck and starboard island. Among the biggest instances by footprint. |
| 7 | **Midway** | Midway-class aircraft carrier (US Navy, now a museum ship). Large flat-deck carrier, somewhat smaller than modern supercarriers. Recognised by its carrier deck layout. |
| 8 | **Ticonderoga** | Ticonderoga-class Aegis guided-missile cruiser (US Navy). A surface combatant with boxy superstructure housing phased-array radars and vertical-launch cells. Smaller than a carrier but larger than most destroyers. |
| 9 | **Other Destroyer** | Destroyer that doesn't match a named destroyer class. Sleek combatant hull with a central superstructure. Fallback within the destroyer (DD) branch. |
| 10 | **Atago DD** | Atago-class Aegis destroyer (Japan Maritime Self-Defense Force). Large guided-missile destroyer with a prominent radar superstructure. Resembles an enlarged Arleigh Burke. |
| 11 | **Arleigh Burke DD** | Arleigh Burke-class Aegis guided-missile destroyer (US Navy), the most numerous modern US surface combatant. Flush-deck hull with a blocky Aegis deckhouse and helicopter facilities. A common warship type in the dataset. |
| 12 | **Hatsuyuki DD** | Hatsuyuki-class destroyer (JMSDF), an older general-purpose escort destroyer. Smaller than modern Aegis ships, with conventional masts and funnels. Recognised by its compact destroyer profile. |
| 13 | **Hyuga DD** | Hyūga-class "helicopter destroyer" (JMSDF) — effectively a flat-deck helicopter carrier classified as a destroyer. Has a full-length flight deck with a starboard island, so it visually resembles a small carrier. Distinct from gun/missile destroyers. |
| 14 | **Asagiri DD** | Asagiri-class destroyer (JMSDF), a general-purpose escort from the late 1980s. Conventional destroyer hull with twin funnels and lattice masts. Mid-sized combatant silhouette. |
| 15 | **Other Frigate** | Frigate not matching a named frigate class. Smaller escort combatant than a destroyer. Fallback within the frigate (FF) branch. |
| 16 | **Perry FF** | Oliver Hazard Perry-class guided-missile frigate. Single-screw escort with a characteristic single large superstructure forward of a long aft deck and helicopter hangar. Widely exported, so it appears in many navies. |
| 17 | **Patrol** | Small patrol craft or patrol boat used for coastal/harbour security. Short hull with a modest superstructure. Among the smaller military instances. |
| 18 | **Other Landing** | Amphibious landing ship not matching a named landing class. Broad, boxy hull designed to carry/deploy troops and vehicles. Fallback within the landing (LL) branch. |
| 19 | **YuTing LL** | Yuting-class tank landing ship (LST, PLA Navy). Bow-ramp amphibious ship with a long open vehicle deck and aft superstructure. Used to land vehicles directly onto a beach. |
| 20 | **YuDeng LL** | Yudeng-class landing ship (PLA Navy), a medium tank/utility landing vessel. Flat, open cargo/vehicle deck with bow doors. Smaller amphibious type. |
| 21 | **YuDao LL** | Yudao-class landing ship (PLA Navy), a medium landing craft. Boxy amphibious hull for shore-to-shore transport. Compact landing-ship profile. |
| 22 | **YuZhao LL** | Yuzhao-class (Type 071) amphibious transport dock (PLA Navy). Large amphibious ship with a flight deck aft and a well dock for landing craft. One of the bigger amphibious silhouettes. |
| 23 | **Austin LL** | Austin-class amphibious transport dock (LPD, US Navy). Combines a helicopter deck with an internal well dock. Large box-like amphibious hull. |
| 24 | **Osumi LL** | Ōsumi-class tank landing ship (JMSDF). Features a full-length flat upper deck and a well dock, giving it a flat-top, carrier-like appearance from above. Used for amphibious and disaster-relief transport. |
| 25 | **Wasp LL** | Wasp-class amphibious assault ship (LHD, US Navy). A large flat-deck "mini-carrier" operating helicopters and STOVL jets, with a well dock. Among the largest amphibious instances. |
| 26 | **LSD 41 LL** | Whidbey Island-class dock landing ship (lead ship LSD-41, US Navy). Built around a large well dock for landing craft, with a smaller superstructure forward. Long, low amphibious hull. |
| 27 | **LHA LL** | Landing Helicopter Assault ship (Tarawa/America-class, US Navy). Large flat-deck amphibious assault ships resembling small aircraft carriers. Operate helicopters and STOVL aircraft. |
| 28 | **Commander** | Command ship serving as a floating headquarters for a fleet or amphibious force. Distinguished by extensive communications antennas and superstructure. Relatively rare class. |
| 29 | **Other Auxiliary Ship** | Non-combatant naval support ship not matching a named auxiliary class. Provides logistics, supply, or service roles. Fallback within the auxiliary (AS) branch. |
| 30 | **Medical Ship** | Hospital ship providing afloat medical care. Often a large merchant-style hull, sometimes marked with red crosses. Long, plain superstructure distinguishes it from cargo vessels. |
| 31 | **Test Ship** | Trials/experimental vessel used to test weapons, sensors, or systems. May carry unusual or non-standard equipment on deck. Uncommon, with a distinctive cluttered topside. |
| 32 | **Training Ship** | Vessel used to train naval cadets or crews. Often a repurposed or purpose-built hull with extra accommodation. Rare class with a utilitarian profile. |
| 33 | **AOE** | Fast Combat Support Ship — an underway-replenishment vessel that resupplies fuel, ammunition, and stores at sea. Large hull lined with cargo-transfer kingposts and replenishment rigs. Supports task groups on the move. |
| 34 | **Masyuu AS** | Mashū-class replenishment oiler (JMSDF), a large fast support/refuelling ship. Long hull with prominent replenishment-at-sea masts amidships. Auxiliary logistics vessel. |
| 35 | **Sanantonio AS** | San Antonio-class amphibious transport dock (US Navy), listed here under the auxiliary branch. Distinctive faceted, stealth-shaped superstructure and twin enclosed mast towers. Carries troops, vehicles, and landing craft. |
| 36 | **EPF** | Expeditionary Fast Transport (Spearhead-class), a high-speed aluminium **catamaran** for rapid intra-theatre transport. Its twin-hull catamaran shape is highly distinctive from above. Used to move troops and cargo quickly. |
| 37 | **Other Merchant** | Commercial/civilian ship not matching a named merchant class. Generic cargo or trade vessel. Fallback within the merchant branch. |
| 38 | **Container Ship** | Cargo ship carrying standardized intermodal containers. Recognised by long open holds stacked with a regular grid of containers and an aft superstructure. Often one of the largest merchant hulls. |
| 39 | **RoRo** | **Roll-on/Roll-off** ship that carries wheeled cargo (cars, trucks, trailers) driven on and off via ramps. Appears as a tall, boxy, slab-sided hull with a continuous flat upper deck and few deck features. Used for vehicle and trailer transport. |
| 40 | **Cargo** | General dry-cargo or bulk freighter. Long hull with open holds/hatch covers and deck cranes, superstructure aft. Carries break-bulk or bulk goods. |
| 41 | **Barge** | Flat-bottomed, usually non-self-propelled vessel for carrying bulk cargo on rivers/harbours. Appears as a simple flat rectangle, often towed or pushed. No (or minimal) superstructure. |
| 42 | **Tugboat** | Small, powerful boat used to tow or push larger ships and maneuver them in port. Compact hull dominated by a central wheelhouse. Frequently seen near docks. |
| 43 | **Ferry** | Passenger and/or vehicle ferry on regular short routes. Boxy multi-deck superstructure spanning most of the hull, often with bow/stern ramps. Distinguished from cargo ships by its passenger decks. |
| 44 | **Yacht** | Recreational/leisure vessel, typically private. Sleek hull with clean lines and open lounging decks. Smaller than commercial ships. |
| 45 | **Sailboat** | Wind-powered recreational craft. Narrow hull with one or more masts; sails or rigging may be visible from above. Among the smallest instances. |
| 46 | **Fishing Vessel** | Commercial fishing boat or trawler. Working hull with nets, booms, or gantries aft. Varies in size but typically small-to-medium. |
| 47 | **Oil Tanker** | Liquid-bulk carrier for oil/petroleum products. Long, flush hull with a maze of deck piping and manifolds and a single aft superstructure. Often very large. |
| 48 | **Hovercraft** | Air-cushion vehicle (e.g. military LCAC) that rides on a cushion of air. Rectangular skirt outline with lift fans and a central cargo deck. Operates between ship and shore. |
| 49 | **Motorboat** | Small powered recreational or utility boat. Short hull with an outboard/inboard engine. One of the smallest classes by footprint. |
| 50 | **Dock** | Dock/floating structure (e.g. dry dock, floating dock, or pier infrastructure) annotated as part of the scene. Not a ship itself, but a fixed/floating harbour structure. Helps the model separate berthing infrastructure from vessels. |

## Notes on selected classes

- **RoRo (Roll-on/Roll-off):** purpose-built for cargo that is *driven* aboard
  rather than lifted. The tall, featureless, slab-sided box shape (to maximise
  internal vehicle decks) makes it visually distinct from container ships, which
  show a stacked-container grid, and from tankers, which show deck piping.
- **"Other …" classes:** every branch of the tree (warship, destroyer, frigate,
  landing, auxiliary, merchant) has an *Other* fallback. These exist so the
  annotation tree stays complete even when a ship's exact class can't be
  determined — important for honest training and evaluation rather than forcing a
  wrong fine-grained label.
- **Flat-deck confusion set:** several classes (aircraft carriers, *Hyuga DD*,
  *Osumi LL*, *Wasp LL*, *LHA LL*) all present a large flat top deck from above
  and are visually similar — a key fine-grained challenge of the dataset.

## References

- Repository: <https://github.com/zzndream/ShipRSImageNet>
- Benchmark framework: MMDetection v2.11.0 (OpenMMLab).
