(define (my-mask2 filename
                  amount
                  wrapmode
                  edgemode
                  radius
                  black)
   (let* ((image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
          (drawable (car (gimp-image-get-active-layer image))))
     (plug-in-edge RUN-NONINTERACTIVE image drawable amount wrapmode edgemode)
     (plug-in-cartoon RUN-NONINTERACTIVE image drawable radius black)
     (gimp-invert drawable)
     (gimp-file-save RUN-NONINTERACTIVE image drawable filename filename)
     (gimp-image-delete image)))
