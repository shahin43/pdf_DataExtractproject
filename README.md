# PDF dataExtractproject
project for extracting content from pdfs/ images. 
Here using OCR for extracting content from images

## Development
### To set up development environment
Make sure you have docker installed <br>
<br>


- Clone the repo
- cd to repository folder

#### run below command to build image, based on Dockerfile pulled from repo 
```bash  docker image build -t ocr_pdf_model . ```
<br>

#### once image is build, then run following command to run container and get into container 
```bash  docker run -d --name ocr_pdf_model_con ocr_pdf_model``` 
<br>
```bash  docker exec -it ocr_pdf_model_con ```
