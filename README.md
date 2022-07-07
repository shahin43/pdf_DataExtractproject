# PDF dataExtractproject Overview
Project for extracting content from pdfs/ images. 
It uses Optical Character Recognition for extracting information from PDFs. Its difficult to extract details from digital invoices or scanned copies for invoices using python libraries like PDFMiner. Libraries that I used for developing this solution is pdf2image (for converting PDF to images), OpenCV (for Image pre-processing) and finally PyTesseract for OCR along with Python.

This projects converts PDFs to page images using pdf2image (for converting PDF to images), later processing the Images using OpenCV and then finally using Tesseract OCR version >4 (using PyTesseract for OCR ) for retrieving details from the PDF images.


## Development 

### To set up development environment
Make sure you have docker installed <br>
<br>

- Clone the repo
- cd to repository folder

run below command to build image, based on Dockerfile pulled from repo 
```  
$ docker image build -t ocr_pdf_model . 
```

once image is build, then run following command to run container 
```
$ docker run -d --name ocr_pdf_model_con ocr_pdf_model
``` 

access the container using below comment
```  
$ docker exec -it ocr_pdf_model_con bash 
```


once in the container, execute below steps to activate python environment and run script
``` 
$ source ../ocrevn/bin/activate. ## activates the env ocrevn
``` 
NOTE : python script, retrieves pdf files from s3bucket. Script takes *s3bucket*, *s3prefix*  as arguments, 
       requires awscli, its part of build, you need to provide respective aws key and secret with IAM permissions to access the bucket, to process the pdfs in s3.
       
command for configuring awscli credentials, then provide AWS KEY & SECRET
``` 
$ aws configure 
``` 

once awscli is configured properly, you can execute the script as below 
```
$ python pdfDataextract.py --s3bucket 'S3 bucket name' --s3prefix 'S3 prefix'
``` 
The script process the pdfs to images, based on number on pages in pdf document. Does preprocessing, text detection and dumps data for each pdf into *extracted_data* folder, generating a .txt file for each pdf document. In production env, extracted content should be pushed into NoSQL db or Elastic search for retriveing easily using an API.

