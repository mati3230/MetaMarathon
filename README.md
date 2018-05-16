# MetaMarathon

## Requirements

* [VirtualBox](https://www.virtualbox.org/wiki/Downloads)
* At least 15 Gb free disk space

## Setup

### Install VirtualBox

* Follow install instructions of [VirtualBox](https://www.virtualbox.org/wiki/Downloads)

### Download Image

* Downloaded .vdi-file from [nextcloud](https://nextcloud.mirevi.medien.hs-duesseldorf.de/index.php/s/OvNqimOvcumJWWa)

### Import Image

* Click **New-Button**
* Type an appropriate name for the image
* Choose **Type Linux** and **Version Ubuntu (64 Bit)**
* If you are not able to choose 64 bit: 
	* Ensure [Hyper-V](https://www.poweronplatforms.com/enable-disable-hyper-v-windows-10-8/) is disabled
	* Ensure [Hardware virtualization is enabled in the BIOS](https://superuser.com/questions/866962/why-does-virtualbox-only-have-32-bit-option-no-64-bit-option-on-windows-7?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)
* Choose **Use an existing virtual hard drive file**
* Click on the folder button and browse for the .vdi file which you downloaded in the previous step
* Click **Create**
![alt text](./images/ImportImage01.jpg "Image import options")

### Start the Image

* Double-Click on image created in previous step
* If error occur: make sure [Hyper-V](https://www.poweronplatforms.com/enable-disable-hyper-v-windows-10-8/) is disabled
* Login as **modalg** with password **12345678**

### Start Eclipse

* Click on **Ubuntu-Button** and search for **eclipse**
* Execute eclipse
* Click **ok** when **workspace dialog** appears
![alt text](./images/StartEclipse01.jpg "Start eclipse")

### Run Test Project

* Open **test** folder in **modalg181-Project**
* Double click on any of the files
* Execute file with click on **run** button (**run as Python Run**)
![alt text](./images/RunScript01.jpg "Run script in eclipse")

### Clone this repository and setup eclipse

* Open explorer
* Navigate to folder **Projekte**
* Right click in explorer and select **Open in terminal** option
* Type **git clone https://github.com/mati3230/MetaMarathon.git** in terminal
* If closed, start eclipse 
* Right click in **PyDev Package Explorer**
* Select **New/Project...** option
![alt text](./images/CreateProject01.jpg "Create Project for MetaMarathon")
* Select **PyDev/PyDev Project** and click **Next**
![alt text](./images/CreateProject03.jpg "Select PyDev Project")
* Type project name **MetaMarathon**
* Select grammar version **3.0-3.5**
* Select interpreter **/home/modalg/miniconda3/envs/modalg/bin/python3.5**
* Press **Finish**
![alt text](./images/CreateProject02.jpg "Setup Project Properties")