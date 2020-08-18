from requests import exceptions
import imutils
import requests
import cv2
import os


def retrieve_images(search_term, api_key, max_res, group_size, save_path):
    """
    retrieves images from bing API to a specific search term
    Afterwards, compatibility to opencv is checked and images are sized down to prevent upload issues to labelme

    reference to: https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/
    Args:
        save_path: directory where the images should be saved
        search_term: term to search for in bing API
        api_key: user key of bing API
        max_res: number of results to query for
        group_size: step size of one request (maximum of 50, prevents timeout)

    Returns:
        None

    """

    # set the link to the api endpoint
    url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

    # excpetions that can appear during querying and saving files
    set_exceptions = set([IOError, FileNotFoundError, exceptions.RequestException,
                          exceptions.HTTPError, exceptions.ConnectionError, exceptions.Timeout])

    # set header and params for http request
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    params = {"q": search_term, "offset": 0, "count": group_size}

    # look up the number of possible results
    print("[INFO] searching Bing API for '{}'".format(search_term))
    search = requests.get(url, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    est_num_results = min(results["totalEstimatedMatches"], max_res)
    print("[INFO] {} total results for '{}'".format(est_num_results, search_term))

    # count images successfully saved so far
    total = 0

    # loop over the expected amount of images
    for offset in range(0, est_num_results, group_size):

        # perform search for the current parameters (esp. offset)
        print("[INFO] making request for group {}-{} of {}...".format(offset, offset+group_size, est_num_results))
        params["offset"] = offset
        search = requests.get(url, headers=headers, params=params)
        search.raise_for_status()
        results = search.json()

        print("[INFO] saving images for group {}-{} of {}...".format(offset, offset+group_size, est_num_results))

        # loop over results
        for v in results["value"]:
            try:  # download the image
                print("[INFO] fetching: {}".format(v["contentUrl"]))
                r = requests.get(v["contentUrl"], timeout=30)

                # build the save path
                ext = v["contentUrl"][v["contentUrl"].rfind("."):]
                p = os.path.sep.join([save_path, "{}_{}{}".format(search_term, str(total).zfill(8), ext)])

                # save image
                f = open(p, "wb")
                f.write(r.content)
                f.close()

            # catch any error that occurs during downloading the images
            except Exception as e:
                if type(e) in set_exceptions:
                    print("[INFO] skipping {}".format(v["contentUrl"]))
                    continue
                else:
                    print(e)

            # try to load the image with opencv
            image = cv2.imread(p)

            # delete image if it is not compatible to opencv
            if image is None:
                try:
                    print("[INFO] deleting: {}".format(p))
                    os.remove(p)
                except FileNotFoundError:
                    continue
                continue
            else:  # resize image for upload to labelme
                width, height = image.shape[:2]
                if width > height:
                    image = imutils.resize(image, width=1400)
                else:
                    image = imutils.resize(image, height=1400)

                cv2.imwrite(p, image)

            total += 1


if __name__ == "__main__":
    retrieve_images("stairs outdoor", "387af050e2a049098f07a9a519943a23", 200, 10, 'img')