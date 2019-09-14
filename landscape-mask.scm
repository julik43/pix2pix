(define (my-mask filename
                  horizontal
                  vertical
                  step
                  exponent)
   (let* ((image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
          (drawable (car (gimp-image-get-active-layer image))))
     (plug-in-gauss RUN-NONINTERACTIVE image drawable horizontal vertical 1)
     (plug-in-oilify RUN-NONINTERACTIVE image drawable step exponent)
     (gimp-file-save RUN-NONINTERACTIVE image drawable filename filename)
     (gimp-image-delete image)))
