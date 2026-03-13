from pathlib import Path
import cv2
import re

class Blade30Dataset:
    def __init__(self, dataset_dir = r"C:\Users\13694\Blade30" , subset: int = 1, read_img = False):
        dataset_dir = Path(dataset_dir)

        self.subset_dir:Path = dataset_dir / f"Blade_{subset}"
        
        # Find the x_x directory under subset_dir
        # Pattern: one or more digits followed by underscore followed by one or more digits
        pattern = re.compile(r'^\d+_\d+$')
        matching_dirs = [d for d in self.subset_dir.iterdir() if d.is_dir() and pattern.match(d.name)]
        
        assert len(matching_dirs) == 1, f"Expected exactly one x_x directory under {self.subset_dir}, found {len(matching_dirs)}"
        self.img_path = matching_dirs[0]
        self.mask_path = self.img_path / "mask"
        
        # Collect image files and load images with masks
        self.images = []
        self.masks = []
        self.colmap_mask_dir = self.subset_dir / "colmap_mask"
        self.colmap_mask_dir.mkdir(exist_ok=1, parents=1)
        
        img_list = list(self.img_path.glob("*.jpg"))
        img_list.sort()
        for img_file in img_list:
            # Read image
            if read_img: 
                img = cv2.imread(str(img_file))
            
            # Read corresponding mask
            mask_file = self.mask_path / f"{img_file.stem}.png"
            resized_mask_file = mask_file.parent/(mask_file.stem + "_resized.png")
            colmap_mask_file = self.colmap_mask_dir / (img_file.name + ".png")
            
            if resized_mask_file.exists():
                resized_mask_file.unlink()
        
            if read_img: 
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
                # Check if mask shape matches image shape
                if not resized_mask_file.exists():
                    cv2.imwrite( str(resized_mask_file) , mask)
                if not colmap_mask_file.exists():
                    cv2.imwrite(str(colmap_mask_file), mask)

            
            self.images.append(img_file.__str__())
            self.masks.append(colmap_mask_file.__str__())
        print(self.subset_dir.name)
        print(f"Image number: {len(self.images)}")
        print(f"Mask number: {len(self.masks)}")

if __name__ == '__main__':
    for subset in range(1,30+1):
        blade = Blade30Dataset(subset=subset)
        del blade
    pass

