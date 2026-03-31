
import xml.etree.ElementTree as ET
import os

URDF_PATH = "../../resources/robots/hustdog2/urdf/hustdog2.urdf"

def fix_urdf():
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()
    
    legs = ['FL', 'FR', 'RL', 'RR']
    
    for leg in legs:
        link_name = f"{leg}_foot"
        joint_name = f"{leg}_foot_fixed"
        
        # Find Link
        link = None
        for l in root.findall('link'):
            if l.get('name') == link_name:
                link = l
                break
        
        # Find Joint
        joint = None
        for j in root.findall('joint'):
            if j.get('name') == joint_name:
                joint = j
                break
                
        if link is None or joint is None:
            print(f"Skipping {leg} (Link or Joint not found)")
            continue
            
        # Get Inertial Origin Y
        inertial = link.find('inertial')
        inertial_origin = inertial.find('origin')
        xyz_inertial = [float(x) for x in inertial_origin.get('xyz').split()]
        y_offset = xyz_inertial[1]
        
        print(f"Fixing {leg}: Found Inertial Offset Y = {y_offset}")
        
        # 1. Update Joint Origin (Move Link Frame to Inertial Center)
        joint_origin = joint.find('origin')
        xyz_joint = [float(x) for x in joint_origin.get('xyz').split()]
        xyz_joint[1] += y_offset
        joint_origin.set('xyz', f"{xyz_joint[0]} {xyz_joint[1]} {xyz_joint[2]}")
        
        # 2. Update Visual Origin (Move Mesh back relative to Link Frame)
        visual = link.find('visual')
        visual_origin = visual.find('origin')
        xyz_visual = [float(x) for x in visual_origin.get('xyz').split()]
        xyz_visual[1] -= y_offset
        visual_origin.set('xyz', f"{xyz_visual[0]} {xyz_visual[1]} {xyz_visual[2]}")
        
        # 3. Update Inertial Origin (Move Inertial Center back relative to Link Frame -> Should be 0)
        xyz_inertial[1] -= y_offset
        inertial_origin.set('xyz', f"{xyz_inertial[0]} {xyz_inertial[1]} {xyz_inertial[2]}")
        
    tree.write(URDF_PATH, encoding="utf-8", xml_declaration=True)
    print("URDF Updated successfully!")

if __name__ == "__main__":
    fix_urdf()
